"""
    `python visualize_clusters.py -h` for usage help
"""
import os
import random
import argparse
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append(os.path.abspath('..'))
from common.image_processing import get_galaxy_pixels, Cluster


data_dir = os.path.join('..', 'data', 'scored')
labels_dir = os.path.join('..', 'data', 'scored.csv')
RED = [255, 0, 0]


def noise_in_background(img, background_threshold):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    # draw original image on the left plot
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original image')
    galaxy_coords, _ = get_galaxy_pixels(img, threshold=background_threshold)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # write textual cluster information next to each cluster
    clusters = Cluster.find_clusters(img, galaxy_coords)
    print("# single point clusters:", len([c for c in clusters if c.size() == 1]))
    print("# intensity 1 clusters:", len([c for c in clusters if c.get_intensity() == 1]))
    print("Num of 0 intensity pixels: ", img[img == 0].size)
    for cluster in clusters:
        centerx, centery = cluster.get_center_pixel()
        text = 'Intensity: {}'.format(cluster.get_intensity())
        cv2.putText(img_rgb, text,
                    (centery - 80, centerx - 20),  # coordinates are reversed here for some ungodly reason
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    1)

    # mark "non-background" spots red
    for coord in galaxy_coords:
        img_rgb[coord[0], coord[1], :] = RED

    ax2.imshow(img_rgb)
    ax2.set_title('All pixels with value >= {}'.format(background_threshold))
    description = "Number of clusters found: {}".format(len(clusters))
    fig.text(.5, .05, description, ha='center')
    plt.show()


def visualize_noise_in_background(n_images, background_threshold, min_score, max_score, randomize=True):
    scored_file_ids = [x.replace('.png', '') for x in os.listdir(data_dir)]
    scores_df = pd.read_csv(labels_dir, index_col='Id')
    assert not scores_df.isnull().values.any()
    scores_df.Actual = scores_df.Actual.astype('float')
    real_images_df = scores_df[(scores_df.Actual >= min_score) & (scores_df.Actual < max_score)]
    real_images_df.index = real_images_df.index.astype(str, copy=False)
    real_image_ids = [image_id for image_id in scored_file_ids if image_id in real_images_df.index.tolist()]

    print("# of real images ", len(real_image_ids))

    if randomize:
        random.shuffle(real_image_ids)

    for ind, image_id in enumerate(real_image_ids):
        if ind == n_images:
            break

        img = cv2.imread(os.path.join(data_dir, '{}.png'.format(image_id)), cv2.IMREAD_GRAYSCALE)

        print(scores_df['Actual'][int(image_id)])
        noise_in_background(img, background_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizes small and almost invisible clusters (possibly '
                                                 'representing galaxies) in images. Picks real galaxy images randomly,'
                                                 'and displays the original image, along with the image where all'
                                                 'the clusters are marked in red.')
    parser.add_argument('--n_images', type=int, default=1, help='number of images to visualize')
    parser.add_argument('--min_score', type=float, default=1.0, help='Min score of images to visualize')
    parser.add_argument('--max_score', type=float, default=2.0, help='Max score of images to visualize')
    parser.add_argument('--background_threshold', type=int, default=10, help='Pixels strictly above given intensity(0-255) '
                                                                             'are considered non-background')
    args = parser.parse_args()

    assert args.min_score < args.max_score

    visualize_noise_in_background(args.n_images,
                                  args.background_threshold,
                                  args.min_score,
                                  args.max_score)
