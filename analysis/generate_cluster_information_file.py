import os
import argparse

import pickle
import cv2
import pandas as pd
from tqdm import tqdm

from baseline.image_processing import Cluster, get_galaxy_pixels, estimate_background_intensity_threshold, pixel_intensity_histogram

main_data_dir = os.path.join('..', 'data')

scored_images_dir = os.path.join('..', 'data', 'scored')
scores_path = os.path.join('..', 'data', 'scored.csv')
scored_images_pkl_out = os.path.join(main_data_dir, 'scored_info.pkl')
scored_images_df_out = os.path.join(main_data_dir, 'scored_info.csv')

labeled_images_dir = os.path.join('..', 'data', 'labeled')
labels_path = os.path.join('..', 'data', 'labeled.csv')
labeled_images_pkl_out = os.path.join(main_data_dir, 'labeled_info.pkl')
labeled_images_df_out = os.path.join(main_data_dir, 'labeled_info.csv')


def extract_image_information(img, background_threshold=None):

    if background_threshold is None:
        background_threshold = estimate_background_intensity_threshold(img, background_pixels_ratio=0.8)

    data = {}
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


def load(format='df'):
    if format == 'pkl':
        with open(scored_images_pkl_out, 'rb') as f:
            data = pickle.load(f)
            return data
    else:
        df = pd.read_csv(scored_images_df_out, index_col='id')
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


def generate_scored_images_info_file(min_score, background_threshold=None):
    scored_image_ids = [x.replace('.png', '') for x in os.listdir(scored_images_dir)]
    scores_df = pd.read_csv(scores_path, index_col='Id')
    scores_df.index = scores_df.index.astype(str)

    if min_score is not None:
        scored_image_ids = scores_df[scores_df.Actual >= min_score].index.values

    image_data = {}
    for ind, image_id in enumerate(tqdm(scored_image_ids)):
        # if ind == 4:
        #     break
        image_path = os.path.join(scored_images_dir, "{}.png".format(image_id))
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        data = extract_image_information(img, background_threshold=background_threshold)
        data['score'] = scores_df.at[image_id, 'Actual']
        image_data[image_id] = data

    if not os.path.exists(main_data_dir):
        os.makedirs(main_data_dir)

    # save as a pickled python dict file
    with open(scored_images_pkl_out, 'wb') as f:
        pickle.dump(image_data, f)

    df = to_df(image_data)
    df.to_csv(scored_images_df_out, index_label='id')


def generate_labeled_images_info_file(background_threshold=4):
    """
    Currently it only considers real images
    @return:
    @rtype:
    """
    labels_df = pd.read_csv(labels_path, index_col='Id')
    labels_df.index = labels_df.index.astype(str)
    labels_df = labels_df[labels_df.Actual == 1.0]

    real_image_ids = labels_df.index.values
    image_data = {}
    for ind, image_id in enumerate(tqdm(real_image_ids)):
        # if ind == 4:
        #     break
        image_path = os.path.join(labeled_images_dir, "{}.png".format(image_id))
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        data = extract_image_information(img, background_threshold=background_threshold)
        image_data[image_id] = data

    # TODO: save as csv as well
    print("Saving pkl to: {}".format(labeled_images_pkl_out))
    print(image_data)
    with open(labeled_images_pkl_out, 'wb') as f:
        pickle.dump(image_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates a file containing galaxy information about all the images from the dataset.')
    parser.add_argument('--dataset', type=str, choices=['scored', 'labeled'], default='scored', help='scored or labeled dataset')
    parser.add_argument('--background_threshold', type=int, default=5, help='minimum pixel intensity (0-255) for a '
                                                                            'pixel to be considered as part of a '
                                                                            'galaxy')
    parser.add_argument('--min_score', type=float, default=None, help='disregard fake images i.e images with score < min_score,'
                                                                      'used only if dataset is scored')
    args = parser.parse_args()

    if args.dataset == 'scored':
        generate_scored_images_info_file(args.min_score, background_threshold=args.background_threshold)
    else:
        generate_labeled_images_info_file(background_threshold=args.background_threshold)