import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils as vutils
from torch import nn
from tqdm import tqdm

class PositionGanModel(nn.Module):
    def __init__(self, weights_file="../pretrained_weights/position_gan_weights", device=torch.device("cpu"), batch_size=8, grid_size=(32, 32)):
        gen_base_filters = 8
        self.latent_dim_position = 16
        self.batch_size = batch_size
        self.device = device
        self.grid_size = grid_size
        super(PositionGanModel, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # latent_dim x 1 x 1
            nn.ConvTranspose2d( self.latent_dim_position, gen_base_filters * 8, 4, 1, 0, bias=False),
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
        print("Galaxy model loaded!")
    
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

class GalaxyGanModel(nn.Module):
    def __init__(self, weights_file, device="cpu", batch_size=8, patch_size=(32, 32)):
        super(GalaxyGanModel, self).__init__()
        self.latent_dim_galaxy = 32
        self.init_size = 32 // 4
        self.patch_size = patch_size

        self.l1 = nn.Sequential(nn.Linear(latent_dim_galaxy, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def load_weights(self, weights_file):
        print("Loading Galaxy model...")
        self.load_state_dict(torch.load(weights_file, map_location=self.device))
        self.eval()
        self.to(self.device)
        print("Galaxy model loaded!")
    
    def generate(self, n_images):
        output = []
        print("Generating galaxies:")
        for i in tqdm(range(0, n_images, self.batch_size)):
            size = min(self.batch_size, n_images - i)
            noise = torch.randn(size, self.latent_dim_galaxy, device=self.device)
            with torch.no_grad():
                galaxies = self(noise).detach().cpu()
            final = torch.clone(galaxies)
            final = (final + 1) / 2
            for j in range(0, size):
                output.append((final[j].numpy().reshape(*self.patch_size)*255).astype(np.uint8))
        return output
    
    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class GalaxyGan(nn.Module):
    def __init__(self, weights_file, device="cpu", batch_size=8, patch_size=(32, 32)):
        super(GalaxyGan, self).__init__()
        gen_base_filters = 64
        self.latent_dim_galaxy = 64
        self.device = device
        self.batch_size = batch_size
        self.patch_size = patch_size

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.latent_dim_galaxy, gen_base_filters * 4, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(gen_base_filters * 4),
            # state size. (gen_base_filters*4) x 4 x 4
            nn.ConvTranspose2d(gen_base_filters * 4, gen_base_filters * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(gen_base_filters * 2),
            # state size. (gen_base_filters*2) x 8 x 8
            nn.ConvTranspose2d( gen_base_filters * 2, gen_base_filters, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(gen_base_filters),
            # state size. (gen_base_filters) x 16 x 16
            nn.ConvTranspose2d( gen_base_filters, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 1 x 32 x 32
        )
        self.load_weights(weights_file)

    def forward(self, input):
        return self.main(input)

    def load_weights(self, weights_file):
        print("Loading Galaxy model...")
        self.load_state_dict(torch.load(weights_file, map_location=self.device))
        model.eval()
        model.to(self.device)
        print("Galaxy model loaded!")
    
    def generate(self, n_images):
        output = []
        print("Generating galaxies:")
        for i in tqdm(range(0, n_images, self.batch_size)):
            size = min(self.batch_size, n_images - i)
            noise = torch.randn(size, self.latent_dim_galaxy, 1, 1, device=self.device)
            with torch.no_grad():
                galaxies = self(noise).detach().cpu()
            final = torch.clone(galaxies)
            final = (final + 1) / 2
            for j in range(0, size):
                output.append((final[j].numpy().reshape(*self.patch_size)*255).astype(np.uint8))
        return output
