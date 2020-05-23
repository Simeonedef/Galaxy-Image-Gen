"""
    `python visualize.py -h` for usage help
"""
import os
import random
import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pandas as pd
import sys
sys.path.append(os.path.abspath('..'))
from common.image_processing import get_galaxy_pixels, Cluster

data_dir = os.path.join('..', 'data', 'scored')
labels_dir = os.path.join('..', 'data', 'scored.csv')
RED = [255, 0, 0]

def find_square(x, y, bins):
    i = 0
    while (not (x >= bins[i] and x < bins[i + 1])):
        i += 1
    
    j = 0
    while (not (y >= bins[j] and y < bins[j + 1])):
        j += 1
    
    return (bins[j], bins[i]), (bins[j + 1], bins[i + 1])

def find_and_highlight_pixels(img, threshold=10):
    galaxy_coords, _ = get_galaxy_pixels(img, threshold=threshold)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    bins = [64*i for i in range(0, 17)]
    # write textual cluster information next to each cluster
    clusters = Cluster.find_clusters(img, galaxy_coords)

    for cluster in clusters:
        centerx, centery = cluster.get_center_pixel()
        left, right = find_square(centerx, centery, bins)
        cv2.rectangle(img_rgb, left, right, RED, 2)

    # mark "non-background" spots red
    for coord in galaxy_coords:
        img_rgb[coord[0], coord[1], :] = RED
    return len(clusters), img_rgb

def noise_in_background(img):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    # draw original image on the left plot
    img = cv2.resize(img, (1024, 1024))
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original image')

    n_clusters, img_highlighted = find_and_highlight_pixels(img, threshold=10)
    new_img = ax2.imshow(img_highlighted)
    ax2.set_title('All pixels with value >= {}'.format(10))
    description = "Number of clusters found: {}".format(n_clusters)
    text = fig.text(.5, .05, description, ha='center')
    slider_axes = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    threshold_slider = Slider(slider_axes, 'Threshold', 3, 255, valinit=10, valstep=1)
    def update(val):
        threshold = threshold_slider.val
        n_clusters, img_highlighted = find_and_highlight_pixels(img, threshold=threshold)
        ax2.set_title('All pixels with value >= {}'.format(threshold))
        text.set_text("Number of clusters found: {}".format(n_clusters))
        new_img.set_data(img_highlighted)
        fig.canvas.draw_idle()
    
    threshold_slider.on_changed(update)    
    plt.show()

def noise_in_background_grid(imgs):
    fig, ax = plt.subplots(nrows=3, ncols=3)
    ax_img_list = []
    for index, img in enumerate(imgs):
        ax0 = index // 3
        ax1 = index % 3
        axis = ax[ax0][ax1]
        n_clusters, img_highlighted = find_and_highlight_pixels(img, threshold=10)
        ax_img_list.append(axis.imshow(img_highlighted))
        axis.axis('off')
    slider_axes = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    threshold_slider = Slider(slider_axes, 'Threshold', 3, 255, valinit=10, valstep=1)
    def update(val):
        threshold = threshold_slider.val
        for index, img in enumerate(imgs):
            n_clusters, img_highlighted = find_and_highlight_pixels(img, threshold=threshold)
            ax_img_list[index].set_data(img_highlighted)
        fig.canvas.draw_idle()
    fig.subplots_adjust(hspace=0.1)
    threshold_slider.on_changed(update)
    plt.show()


def visualize_noise_in_background(score_threshold=2.0, randomize=True, grid=False):
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

    if grid:
        imgs = []
        for ind, image_id in enumerate(real_image_ids[:9]):
            imgs.append(cv2.imread(os.path.join(data_dir, '{}.png'.format(image_id)), cv2.IMREAD_GRAYSCALE))
        noise_in_background_grid(imgs)
    else:
        img = cv2.imread(os.path.join(data_dir, '{}.png'.format(real_image_ids[0])), cv2.IMREAD_GRAYSCALE)
        nose_in_background(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizes small and almost invisible clusters (possibly '
                                                 'representing galaxies) in images. Picks real galaxy images randomly,'
                                                 'and displays the original image, along with the image where all'
                                                 'the clusters are marked in red.')
    parser.add_argument('--grid', help='visualize grid', action="store_true")
    args = parser.parse_args()

    visualize_noise_in_background(grid=args.grid)
    # noise_in_background(img)