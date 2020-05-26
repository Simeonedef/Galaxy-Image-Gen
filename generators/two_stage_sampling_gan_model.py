import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch import nn
from tqdm import tqdm

latent_dim_position = 16
latent_dim_galaxy = 32
patch_size = (32, 32)
grid_size = (1024//patch_size[0], 1024//patch_size[1])
batch_size = 8
class TwoStageModel:
    """
    A generative model based on first sampling the locations of galaxies and then generating them
    - The positions of galaxies are generated from the position_gan as a binary grid
    - Places with no galaxies are left black, places with galaxies receive them from 
    the gan that generates galaxies
    - The grid is thus expanded to a 1024*1024 image, which is resized (cropped?) to 1000*1000
    """
    def __init__(self, image_width=1000, image_height=1000, device="cpu",
                 position_gan_weights_file="../pretrained_weights/position_gan_weights",
                 galaxy_gan_weights_file="../pretrained_weights/32_latent32_500epochs_smallThresholds_G.pts"):
        self.image_width = image_width
        self.image_height = image_height
        self.device = torch.device(device)
        self.position_gan = self.load_position_gan(position_gan_weights_file)
        self.galaxy_gan = self.load_galaxy_gan(galaxy_gan_weights_file)

    def generate(self, n_images):
        positions, n_galaxies = self.generate_positions(n_images)
        galaxies = self.generate_galaxies(n_galaxies)
        cur_galaxy = 0
        imgs = []
        for p in positions:
            new_img = np.zeros((1024, 1024))

            p = list(zip(*np.where(p==1)))
            for location in p:
                new_img[int(location[0]*patch_size[0]):int((location[0]+1)*patch_size[0]), int(location[1]*patch_size[1]):int((location[1]+1)*patch_size[1])] = galaxies[cur_galaxy]
                cur_galaxy += 1
            # crop for now, maybe in the future resize
            imgs.append(new_img[:self.image_height, :self.image_width])
        return imgs

    def load_position_gan(self, weights_file):
        print("Loading position model...")
        model = PositionGanModel()
        model.load_state_dict(torch.load(weights_file, map_location=self.device))
        model.eval()
        model.to(self.device)
        print("Position model loaded!")
        return model
    
    def load_galaxy_gan(self, weights_file):
        print("Loading Galaxy model...")
        model = GalaxyGanModel()
        model.load_state_dict(torch.load(weights_file, map_location=self.device))
        model.eval()
        model.to(self.device)
        print("Galaxy model loaded!")
        return model
    
    def generate_positions(self, n_images):
        output = []
        n_galaxies = 0
        print("Generating positions:")
        for i in tqdm(range(0, n_images, batch_size)):
            size = min(batch_size, n_images - i)
            noise = torch.randn(size, latent_dim_position, 1, 1, device=self.device)
            with torch.no_grad():
                positions = self.position_gan(noise).detach().cpu()
            final = torch.clone(positions)
            final[final <= 0] = 0
            final[final > 0] = 1
            n_galaxies += int(final.sum())
            for j in range(0, size):
                output.append(final[j].numpy().reshape(*grid_size))
        return output, n_galaxies

    def generate_galaxies(self, n_images):
        output = []
        print("Generating galaxies:")
        for i in tqdm(range(0, n_images, batch_size)):
            size = min(batch_size, n_images - i)
            noise = torch.Tensor(np.random.normal(0, 1, (size, latent_dim_galaxy)))
            with torch.no_grad():
                galaxies = self.galaxy_gan(noise).detach().cpu()
            final = torch.clone(galaxies)
            final = (final + 1) / 2
            for j in range(0, size):
                output.append((final[j].numpy().reshape(*patch_size)*255).astype(np.uint8))
        return output

class PositionGanModel(nn.Module):
    def __init__(self):
        gen_base_filters = 8
        super(PositionGanModel, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # latent_dim x 1 x 1
            nn.ConvTranspose2d( latent_dim_position, gen_base_filters * 8, 4, 1, 0, bias=False),
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

    def forward(self, input):
        return self.main(input)

class GalaxyGanModel(nn.Module):
    def __init__(self):
        super(GalaxyGanModel, self).__init__()

        self.init_size = patch_size[0] // 4
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

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


if __name__ == "__main__":
    model = TwoStageModel()
    imgs = model.generate(16)
    plt.imshow(imgs[0], cmap="gray")
    plt.show()