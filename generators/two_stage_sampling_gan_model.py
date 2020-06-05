import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils as vutils
from torch import nn
from tqdm import tqdm
from galaxy_patch_generators import GalaxyGanFirst, GalaxyGanConv
from gan_position_generators import PositionGanModel

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
    def __init__(self, image_width=1000, image_height=1000, device="cpu"):
        self.image_width = image_width
        self.image_height = image_height
        self.device = torch.device(device)
        self.position_gan = PositionGanModel(device=self.device)
        self.galaxy_gan = GalaxyGanFirst(device=self.device)

    def generate(self, n_images):
        positions, n_galaxies = self.position_gan.generate(n_images)
        galaxies = self.galaxy_gan.generate(n_galaxies)
        cur_galaxy = 0
        imgs = []
        for p in positions:
            new_img = np.zeros((1024, 1024))

            p = list(zip(*np.where(p==1)))
            for location in p:
                top = int(location[0]*patch_size[0])
                bottom = int((location[0]+1)*patch_size[0])
                left = int(location[1]*patch_size[1])
                right = int((location[1]+1)*patch_size[1])
                new_img[top:bottom, left:right] = galaxies[cur_galaxy]
                cur_galaxy += 1

            # add small galaxies to new_img
            # crop for now, maybe in the future resize
            imgs.append(new_img[:self.image_height, :self.image_width])
        return imgs

if __name__ == "__main__":
    model = TwoStageModel()
    imgs = model.generate(9)
    grid = vutils.make_grid(torch.Tensor(imgs).reshape(9, 1, 1000, 1000), padding=10, pad_value=1, normalize=True, range=(0, 255), nrow=3)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()