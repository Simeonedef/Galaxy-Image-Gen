import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from generators.full_galaxy_gan_generator import FullGalaxyGan
from generators.two_stage_sampling_gan_model import TwoStageModel, TwoStageConditionalModel
from generators.baseline_2 import SmallLargeClustersGenerativeModel
from generators.galaxy_patch_generators import GalaxyGanConditional, GalaxyGanConv
import torchvision.utils as vutils
import torch

galaxies_directory = '../data/labeled'
results_dir = '../data/full_galaxies_methods'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


baseline = SmallLargeClustersGenerativeModel(1000, 1000)
full_galaxy = FullGalaxyGan()
two_stage = TwoStageModel()
two_stage_conditional = TwoStageConditionalModel()
n_samples = 20
images_per_row = 5
for i in range(n_samples):
    images = []
    images = [np.asarray(Image.open(os.path.join(galaxies_directory, img)).convert('L')) for img in os.listdir(galaxies_directory)[i*images_per_row:(i + 1) * images_per_row]]

    images += baseline.generate(images_per_row)
    images += full_galaxy.generate(images_per_row)
    images += two_stage.generate(images_per_row)
    images += two_stage_conditional.generate(images_per_row)

    images_1k = [torch.Tensor(img).reshape(1, 1000, 1000)/255. for img in images]
    vutils.save_image(images_1k,  results_dir + "/" + str(i) + ".png", padding=6, pad_value=1, normalize=False, nrow=images_per_row)

    images_256 = [torch.Tensor(cv2.resize(img, (256, 256))).reshape(1, 256, 256)/255. for img in images]
    vutils.save_image(images_256,  results_dir + "/" + str(i) + "_256.png", padding=2, pad_value=1, normalize=False, nrow=images_per_row)

# galaxy patches

galaxies_directory = '../data/labeled_patches_t5'
results_dir = '../data/galaxy_patches_methods'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

unconditional = GalaxyGanConv()
conditional = GalaxyGanConditional()

n_samples = 20
images_per_row = 10
for i in range(n_samples):
    images = []
    images = [np.asarray(Image.open(os.path.join(galaxies_directory, img)).convert('L')) for img in os.listdir(galaxies_directory)[i * images_per_row: (i + 1) * images_per_row]]

    images += unconditional.generate(images_per_row)
    images += conditional.generate(images_per_row, [0]*images_per_row)
    images += conditional.generate(images_per_row, [1]*images_per_row)
    images += conditional.generate(images_per_row, [2]*images_per_row)

    images = [torch.Tensor(img).reshape(1, 32, 32)/255. for img in images]
    vutils.save_image(images,  results_dir + "/" + str(i) + ".png", padding=2, pad_value=1, normalize=False, nrow=images_per_row)

