import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_dir = os.path.join('..', 'data', 'scored')


def get_galaxies(img, threshold=40):
    # find all pixels which are not pitch black
    print(np.where(img != 0))
    print(img.shape)
    galaxy_x, galaxy_y = tuple(np.where(img > threshold))

    # list of pairs representing the galaxy pixel coords
    galaxy_coords = list(zip(galaxy_x, galaxy_y))

    galaxy_pixel_values = img[galaxy_x, galaxy_y]

    return galaxy_coords, galaxy_pixel_values


def get_background(img):
    background_x, background_y = np.where(img == 0)

    return background_x, background_y


def generate_statistics(data_dir, n_images=10):
    for ind, filename in enumerate(os.listdir(data_dir)):
        if ind == n_images:
            break
        img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)

        galaxy_coords, galaxy_pixel_values = get_galaxies(img)
        background_x, background_y = get_background(img)
        num_background_pixels = len(background_x)

        if len(galaxy_coords) > 10000:
            print(galaxy_pixel_values)
            print(img)

        print(filename, len(galaxy_coords), num_background_pixels)


if __name__ == "__main__":
    #generate_statistics(data_dir, n_images=10)
    img = cv2.imread(os.path.join(data_dir, '1010772.png'), cv2.IMREAD_GRAYSCALE)
    show_potential_galaxies(img)