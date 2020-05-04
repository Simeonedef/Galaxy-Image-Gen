import os

import pickle
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from baseline.image_processing import Cluster, get_galaxy_pixels, estimate_background_intensity_threshold, pixel_intensity_histogram

data_dir = os.path.join('../..', 'data', 'scored')
labels_dir = os.path.join('../..', 'data', 'scored.csv')
analysis_dir = os.path.join('../..', 'analysis')

if __name__ == "__main__":
    scored_image_ids = [x.replace('.png', '') for x in os.listdir(data_dir)]
    scores_df = pd.read_csv(labels_dir, index_col='Id')
    scores_df.index = scores_df.index.astype(str)

    # will hold all extracted information
    image_data = {}
    for ind, image_id in enumerate(tqdm(scored_image_ids)):
        if ind == 20:
            break
        data = {}
        image_path = os.path.join(data_dir, "{}.png".format(image_id))
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        background_threshold = estimate_background_intensity_threshold(img, background_pixels_ratio=0.8)
        pixel_intensity_hist = pixel_intensity_histogram(img)
        galaxy_pixels, _ = get_galaxy_pixels(img, threshold=background_threshold)
        clusters = Cluster.find_clusters(img, galaxy_pixels)
        score = scores_df.at[image_id, 'Actual']
        data['score'] = score
        data['background_threshold'] = background_threshold
        data['pixel_intensity_hist'] = pixel_intensity_hist

        image_data[image_id] = data

    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    image_information_file_path = os.path.join(analysis_dir, 'image_data.pkl')
    with open(image_information_file_path, 'wb') as f:
        pickle.dump(image_data, f)

        # print("Num of galaxy coords: ", len(galaxy_coords))
        # if len(galaxy_coords) > 100000:
        #     plt.imshow(img, cmap='gray')
        #     plt.show()
