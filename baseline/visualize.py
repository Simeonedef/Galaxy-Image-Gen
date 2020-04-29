import os
import random
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from image_processing import get_galaxies


data_dir = os.path.join('..', 'data', 'scored')
labels_dir = os.path.join('..', 'data', 'scored.csv')
RED = [255, 0, 0]


def noise_in_background(img):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    # draw original image on the left plot
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original image')
    background_threshold = 5
    galaxy_coords, _ = get_galaxies(img, threshold=background_threshold)

    # mark "background" spots red
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for coord in galaxy_coords:
        img[coord[0], coord[1], :] = RED

    ax2.imshow(img)
    ax1.set_title('All pixels with value >= {}'.format(background_threshold))
    description = "description goes here"
    fig.text(.5, .05, description, ha='center')
    plt.show()


def visualize_noise_in_background(score_threshold=2.0, randomize=True, n_images=5):
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
        noise_in_background(img)


if __name__ == "__main__":
    visualize_noise_in_background(n_images=3)
    # img = cv2.imread(os.path.join(data_dir, '1010772.png'), cv2.IMREAD_GRAYSCALE)
    # noise_in_background(img)