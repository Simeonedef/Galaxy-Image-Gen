import os
import random
import argparse
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from image_processing import get_galaxies, Cluster


data_dir = os.path.join('..', 'data', 'scored')
labels_dir = os.path.join('..', 'data', 'scored.csv')
RED = [255, 0, 0]


def noise_in_background(img, background_threshold):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    # draw original image on the left plot
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original image')
    galaxy_coords, _ = get_galaxies(img, threshold=background_threshold)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # write textual cluster information next to each cluster
    clusters = Cluster.find_clusters(img, galaxy_coords)
    for cluster in clusters:
        centerx, centery = cluster.get_center_pixel()
        print("Cluster center pixel: ", cluster.get_center_pixel())
        print("Cluster size: ", cluster.size())
        print("Cluster intensity: ", cluster.get_intensity())
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


def visualize_noise_in_background(score_threshold=2.0, randomize=True, n_images=5, background_threshold=10):
    scored_file_ids = [x.replace('.png', '') for x in os.listdir(data_dir)]
    scores_df = pd.read_csv(labels_dir, index_col='Id')
    assert not scores_df.isnull().values.any()
    scores_df.Actual = scores_df.Actual.astype('float')
    real_images_df = scores_df[scores_df.Actual >= score_threshold]
    real_images_df.index = real_images_df.index.astype(str, copy=False)
    real_image_ids = [image_id for image_id in scored_file_ids if image_id in real_images_df.index.tolist()]

    print("# of real images ", len(real_image_ids))

    if randomize:
        random.shuffle(real_image_ids)

    for ind, image_id in enumerate(real_image_ids):
        if ind == n_images:
            break

        img = cv2.imread(os.path.join(data_dir, '{}.png'.format(image_id)), cv2.IMREAD_GRAYSCALE)
        noise_in_background(img, background_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizes small and almost invisible clusters (possibly '
                                                 'representing galaxies) in images')
    parser.add_argument('--n_images', type=int, default=1, help='number of images to visualize')
    parser.add_argument('--background_threshold', type=int, default=10, help='minimum pixel intensity (0-255) for a '
                                                                             'pixel to be considered non-background')
    args = parser.parse_args()

    visualize_noise_in_background(n_images=args.n_images, background_threshold=args.background_threshold)
    # noise_in_background(img)