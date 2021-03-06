import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils as vutils
from torch import nn
from tqdm import tqdm

class GalaxyGanFirst(nn.Module):
    def __init__(self, weights_file="../pretrained_weights/32_latent32_500epochs_smallThresholds_G.pts", device=torch.device("cpu"), batch_size=8, patch_size=(32, 32)):
        super(GalaxyGanFirst, self).__init__()
        self.latent_dim_galaxy = 32
        self.init_size = 32 // 4
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.device = device

        self.l1 = nn.Sequential(nn.Linear(self.latent_dim_galaxy, 128 * self.init_size ** 2))
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
        self.load_weights(weights_file)

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

class GalaxyGanConv(nn.Module):
    def __init__(self, weights_file="../pretrained_weights/galaxy_gan_2.6_t_5_500e", device=torch.device("cpu"), batch_size=8, patch_size=(32, 32)):
        super(GalaxyGanConv, self).__init__()
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
        self.eval()
        self.to(self.device)
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

class GalaxyGanConditional(nn.Module):
    def __init__(self, weights_file="../pretrained_weights/galaxy_cgan_size", device=torch.device("cpu"), batch_size=8, patch_size=(32, 32)):
        super(GalaxyGanConditional, self).__init__()
        gen_base_filters = 64
        self.latent_dim = 100
        self.device = device
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.n_classes = 3
        self.label = nn.Sequential(
            nn.Embedding(self.n_classes, 50),
            nn.Linear(50, 8*8)
        )
        # self.latent_conv = nn.ConvTranspose2d( latent_dim, gen_base_filters * 4, 4, 1, 0, bias=False)
        self.latent = nn.Sequential(
            nn.Linear(self.latent_dim, 256*8*8),
            nn.BatchNorm1d(256*8*8),
            nn.LeakyReLU(0.2)
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(257, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(gen_base_filters * 2),
            # state size. (gen_base_filters*2) x 8 x 8
            nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(gen_base_filters),
            # state size. (gen_base_filters) x 16 x 16
            nn.ConvTranspose2d( 64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )
        self.load_weights(weights_file)

    def forward(self, noise, label):
        label_filter = self.label(label)
        label_filter = label_filter.view(-1, 1, 8, 8)
        noise = self.latent(noise)
        # noise = self.latent_bn(noise)
        noise = noise.view(-1, 256, 8, 8)
        x = torch.cat([noise, label_filter], 1)
        return self.main(x)

    def load_weights(self, weights_file):
        print("Loading Galaxy model...")
        self.load_state_dict(torch.load(weights_file, map_location=self.device))
        self.eval()
        self.to(self.device)
        print("Galaxy model loaded!")
    
    def generate(self, n_images, intensities):
        output = []
        print("Generating galaxies:")
        for i in tqdm(range(0, n_images, self.batch_size)):
            size = min(self.batch_size, n_images - i)
            curr_intensities = torch.LongTensor([intensities[j] for j in range(i, i + size)]).to(self.device)
            noise = torch.randn(size, self.latent_dim, device=self.device)
            with torch.no_grad():
                galaxies = self(noise, curr_intensities).detach().cpu()
            final = torch.clone(galaxies)
            final = (final + 1) / 2
            for j in range(0, size):
                output.append((final[j].numpy().reshape(*self.patch_size)*255).astype(np.uint8))
        return output

if __name__ == "__main__":
    gen = GalaxyGanConditional()
    imgs = gen.generate(4, [2, 2, 2, 2])
    print(imgs[0].shape)
    plt.imshow(imgs[0], cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(imgs[1], cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(imgs[2], cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(imgs[3], cmap='gray', vmin=0, vmax=255)
    plt.show()