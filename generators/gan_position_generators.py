import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils as vutils
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import os
import pickle

class PositionGanModel(nn.Module):
    def __init__(self, weights_file="../pretrained_weights/position_gan_2_t_5_1000e_l100_f16", device=torch.device("cpu"), batch_size=8, grid_size=(32, 32)):
        gen_base_filters = 16
        self.latent_dim_position = 100
        self.batch_size = batch_size
        self.device = device
        self.grid_size = grid_size
        super(PositionGanModel, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # latent_dim x 1 x 1
            nn.ConvTranspose2d(self.latent_dim_position, gen_base_filters * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(gen_base_filters * 8),
            # state size. (gen_base_filters*8) x 4 x 4
            nn.ConvTranspose2d(gen_base_filters * 8, gen_base_filters * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(gen_base_filters * 4),
            # state size. (gen_base_filters*4) x 8 x 8
            nn.ConvTranspose2d(gen_base_filters * 4, gen_base_filters * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(gen_base_filters * 2),
            # state size. (gen_base_filters*4) x 16 x 16
            nn.ConvTranspose2d( gen_base_filters * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 1 x 32 x 32
        )
        self.load_weights(weights_file)
    
    def forward(self, input):
        return self.main(input)

    def load_weights(self, weights_file):
        print("Loading Position model...")
        self.load_state_dict(torch.load(weights_file, map_location=self.device))
        self.eval()
        self.to(self.device)
        print("Position model loaded!")
    
    def generate(self, n_images):
        output = []
        n_galaxies = 0
        print("Generating positions:")
        for i in tqdm(range(0, n_images, self.batch_size)):
            size = min(self.batch_size, n_images - i)
            noise = torch.randn(size, self.latent_dim_position, 1, 1, device=self.device)
            with torch.no_grad():
                positions = self(noise).detach().cpu()
            final = torch.clone(positions)
            final[final <= 0] = 0
            final[final > 0] = 1
            n_galaxies += int(final.sum())
            for j in range(0, size):
                output.append(final[j].numpy().reshape(*self.grid_size))
        return output, n_galaxies

class PositionGanConditional(nn.Module):
    def __init__(self, weights_file="../pretrained_weights/position_cgan_size", device=torch.device("cpu"), batch_size=8, grid_size=(32, 32)):
        super(PositionGanConditional, self).__init__()
        self.latent_dim = 100
        self.batch_size = batch_size
        self.device = device
        self.grid_size = grid_size
        gen_base_filters = 32
        n_classes = 4
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # latent_dim x 1 x 1
            nn.ConvTranspose2d( self.latent_dim, gen_base_filters * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(gen_base_filters * 8),
            # state size. (gen_base_filters*8) x 4 x 4
            nn.ConvTranspose2d(gen_base_filters * 8, gen_base_filters * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(gen_base_filters * 4),
            # state size. (gen_base_filters*4) x 8 x 8
            nn.ConvTranspose2d(gen_base_filters * 4, gen_base_filters * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(gen_base_filters * 2),
            # state size. (gen_base_filters*4) x 16 x 16
            nn.ConvTranspose2d( gen_base_filters * 2, n_classes, 4, 2, 1, bias=False)
            # state size. 1 x 32 x 32
        )
        self.load_weights(weights_file)

    def forward(self, input):
        return self.main(input)

    def load_weights(self, weights_file):
        print("Loading Position model...")
        self.load_state_dict(torch.load(weights_file, map_location=self.device))
        self.eval()
        self.to(self.device)
        print("Position model loaded!")
    
    def generate(self, n_images):
        output = []
        n_galaxies = 0
        print("Generating positions:")
        for i in tqdm(range(0, n_images, self.batch_size)):
            size = min(self.batch_size, n_images - i)
            noise = torch.randn(size, self.latent_dim, 1, 1, device=self.device)
            with torch.no_grad():
                positions = self(noise).detach().cpu()
            final = F.softmax(positions, dim=1)
            final = torch.argmax(final, axis=1)
            n_galaxies += (final != 0).sum().item()
            for j in range(0, size):
                output.append(final[j].numpy().reshape(*self.grid_size))
        return output, n_galaxies


class PositionsFromExistingImages:
    '''
    Sanity check: Extracts positions and sizes of galaxies from 
    existing images in the dataset
    '''

    def __init__(self, size_thresholds=[50, 80, 110]):
        self.size_thresholds = size_thresholds
        self.n_classes = len(size_thresholds) + 1 + 1

    def generate(self, n_images):
        labeled_data = "labeled_info.pkl"
        with open(os.path.join("../data/", labeled_data), 'rb') as pickle_file:
          labeled_data = pickle.load(pickle_file)
        centers = {}
        intensities = {}
        sizes = {}
        for id in labeled_data.keys():
          centers[id] = labeled_data[id]['cluster_centers']
          intensities[id] = labeled_data[id]['cluster_peak_intensities']
          sizes[id] = labeled_data[id]['cluster_sizes']
        centers = list(centers.values())
        intensities = list(intensities.values())
        sizes = list(sizes.values())

        output = []
        n_galaxies = 0
        buckets_y = [i*32 for i in range (1024//32)]
        buckets_x = [i*32 for i in range (1024//32)]
        for i in range(n_images):
            image = centers[i]
            image_intensities = intensities[i]
            image_sizes = sizes[i]
            im = np.zeros((self.n_classes, 1024//32, 1024//32), dtype=float)
            im[0, :, :] = 1
            for center_index, center in enumerate(image):
                intensity = image_intensities[center_index]
                size = image_sizes[center_index]
                if intensity < 30:
                    continue
                y = np.digitize(center[0], buckets_y) - 1
                x = np.digitize(center[1], buckets_x) - 1
                size = np.digitize(size, self.size_thresholds, right=True) + 1 # add one since 0 is taken
                im[size, y, x] = 1
                im[0, y, x] = 0
                n_galaxies += 1
            output.append(im)
        
        return output, n_galaxies
if __name__ == "__main__":
    gen = PositionsFromExistingImages()
    imgs, n_galaxies = gen.generate(20)
    print(n_galaxies)
    print(imgs[0].shape)
    print([(img.min(), img.max()) for img in imgs])