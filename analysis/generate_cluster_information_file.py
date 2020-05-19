import os

import pickle
import cv2
import pandas as pd
from tqdm import tqdm

from baseline.image_processing import Cluster, get_galaxy_pixels, estimate_background_intensity_threshold, pixel_intensity_histogram

data_dir = os.path.join('..', 'data', 'scored')
labels_dir = os.path.join('..', 'data', 'scored.csv')
analysis_dir = os.path.join('.', 'analysis')
image_information_file_path = os.path.join(analysis_dir, 'image_data.pkl')
image_information_file_path_df = os.path.join(analysis_dir, 'image_data.csv')


def extract_image_information(img):
    data = {}
    background_threshold = estimate_background_intensity_threshold(img, background_pixels_ratio=0.8)
    # pixel_intensity_hist = pixel_intensity_histogram(img)
    galaxy_pixels, _ = get_galaxy_pixels(img, threshold=background_threshold)
    clusters = Cluster.find_clusters(img, galaxy_pixels)
    data['background_threshold'] = background_threshold
    # data['pixel_intensity_hist'] = pixel_intensity_hist
    data.update({
        'cluster_num': len(clusters),
        'cluster_sizes': [cluster.size() for cluster in clusters],
        'cluster_peak_intensities': [cluster.get_intensity() for cluster in clusters],
        'cluster_num_intensities': [cluster.num_intensities() for cluster in clusters],
        'cluster_centers': [cluster.get_center_pixel() for cluster in clusters]
    })
    return data


def extract_all_information_query(images_dir):
    image_data = {}
    scored_image_ids = [x.replace('.png', '') for x in os.listdir(images_dir)]
    for ind, image_id in enumerate(tqdm(scored_image_ids)):
        # if ind == 4:
        #     break
        image_path = os.path.join(images_dir, "{}.png".format(image_id))
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        data = extract_image_information(img)
        image_data[image_id] = data

    return image_data


def extract_all_information(images_dir):
    image_data = {}
    for ind, image_id in enumerate(tqdm(scored_image_ids)):
        # if ind == 4:
        #     break
        image_path = os.path.join(images_dir, "{}.png".format(image_id))
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        data = extract_image_information(img)
        data['score'] = scores_df.at[image_id, 'Actual']
        image_data[image_id] = data

    return image_data


def load(format='df'):
    if format == 'pkl':
        with open(image_information_file_path, 'rb') as f:
            data = pickle.load(f)
            return data
    else:
        df = pd.read_csv(image_information_file_path_df, index_col='id')
        return df


# util function
def get_values_for(data, key):
    return [data[key] for data in data.values()]


def to_df_query(data):
    df = pd.DataFrame(index=list(data.keys()))
    df.index.name = 'Id'

    df['background_threshold'] = get_values_for(data, 'background_threshold')
    df.background_threshold = df.background_threshold.astype(int)

    df['cluster_num'] = get_values_for(data, 'cluster_num')
    df.cluster_num = df.cluster_num.astype(int)

    df['cluster_sizes'] = get_values_for(data, 'cluster_sizes')
    df['cluster_peak_intensities'] = get_values_for(data, 'cluster_peak_intensities')
    df['cluster_num_intensities'] = get_values_for(data, 'cluster_num_intensities')
    df['cluster_centers'] = get_values_for(data, 'cluster_centers')

    return df


def to_df(data):
    df = pd.DataFrame([data['score'] for data in data.values()],
                      columns=['score'],
                      index=list(data.keys()))
    df.score = df.score.astype('float')

    df['background_threshold'] = get_values_for(data, 'background_threshold')
    df.background_threshold = df.background_threshold.astype(int)

    df['cluster_num'] = get_values_for(data, 'cluster_num')
    df.cluster_num = df.cluster_num.astype(int)

    df['cluster_sizes'] = get_values_for(data, 'cluster_sizes')
    df['cluster_peak_intensities'] = get_values_for(data, 'cluster_peak_intensities')
    df['cluster_num_intensities'] = get_values_for(data, 'cluster_num_intensities')
    df['cluster_centers'] = get_values_for(data, 'cluster_centers')

    return df


if __name__ == "__main__":
    scored_image_ids = [x.replace('.png', '') for x in os.listdir(data_dir)]
    scores_df = pd.read_csv(labels_dir, index_col='Id')
    scores_df.index = scores_df.index.astype(str)

    data = extract_all_information(data_dir)

    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # save as a pickled python dict file
    with open(image_information_file_path, 'wb') as f:
        pickle.dump(data, f)

    # save as csv
    df = to_df(data)
    df.to_csv(image_information_file_path_df, index_label='id')