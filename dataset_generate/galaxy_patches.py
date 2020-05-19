import os
import argparse
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# This is a hack to find the 'baseline' folder, for some reason my PYTHONPATH is not working properly
import sys
sys.path.append(os.path.abspath('..'))
from baseline.image_processing import Cluster, get_galaxy_pixels

labeled_data_dir = os.path.join('..', 'data', 'labeled')
labels_dir = os.path.join('..', 'data', 'labeled.csv')

labeled_data_out_dir = os.path.join('..', 'data', 'labeled_patches')
labels_out_csv = os.path.join('..', 'data', 'labeled_patches.csv')


def labeled_images_galaxy_patches(background_threshold=10,
                                  patch_size=32,
                                  min_galaxy_size=None):

    labels_df = pd.read_csv(labels_dir, index_col='Id')
    labels_df.index = labels_df.index.astype(str)
    labels_df = labels_df[labels_df.Actual == 1.0]

    real_image_ids = labels_df.index.values
    if not os.path.exists(labeled_data_out_dir):
        os.makedirs(labeled_data_out_dir)

    print("Saving images to: {}".format(labeled_data_out_dir))

    # list of dicts
    patch_info = []
    for ind, image_id in enumerate(tqdm(real_image_ids)):
        if ind == 100:
            break
        image_path = os.path.join(labeled_data_dir, "{}.png".format(image_id))
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        galaxy_pixels, _ = get_galaxy_pixels(img, threshold=background_threshold)
        clusters = Cluster.find_clusters(img, galaxy_pixels)

        for cluster_ind, cluster in enumerate(clusters):
            patch = cluster.crop_patch(patch_size)

            # patch goes outside the frame of the image, just ignore it
            if patch is None:
                continue

            patch_id = "{}_{:0>2d}.png".format(image_id, cluster_ind)
            patch_path = os.path.join(labeled_data_out_dir, patch_id)
            plt.imsave(patch_path, patch, cmap='gray')

            curr_patch_info = {'patch_id': patch_id,
                               'intensity': cluster.get_intensity(),
                               'size': cluster.size()}
            patch_info.append(curr_patch_info)

    patch_info_df = to_df(patch_info)
    patch_info_df.to_csv(labels_out_csv)
    print("Saving csv to: {}".format(labels_out_csv))


def to_df(data):
    df = pd.DataFrame(data)
    df.set_index('patch_id', inplace=True)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finds the galaxies in the images, and stores the cropped galaxies ('
                                                 'patches) as independent images.')
    parser.add_argument('--background_threshold', type=int, default=5, help='minimum pixel intensity (0-255) for a '
                                                                             'pixel to be considered as part of a galaxy')
    parser.add_argument('--patch_size', type=int, default=32, help='The size of the extracted galaxy patches')
    parser.add_argument('--min_galaxy_size', type=int, default=None, help='disregard galaxies with number of pixels smaller than this number')
    args = parser.parse_args()

    print("Config: ", args)
    labeled_images_galaxy_patches(background_threshold=args.background_threshold,
                                  patch_size=args.patch_size,
                                  min_galaxy_size=args.min_galaxy_size)