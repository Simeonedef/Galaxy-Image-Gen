import os
import argparse
import sys
import pickle
import cv2
import pandas as pd
from tqdm import tqdm

from common.image_processing import Cluster, get_galaxy_pixels, estimate_background_intensity_threshold, num_background_pixels

main_data_dir = os.path.join('..', 'data')

scored_images_dir = os.path.join('..', 'data', 'scored')
scores_path = os.path.join('..', 'data', 'scored.csv')
scored_images_pkl_out = os.path.join(main_data_dir, 'scored_info.pkl')
scored_images_df_out = os.path.join(main_data_dir, 'scored_info.csv')

labeled_images_dir = os.path.join('..', 'data', 'labeled')
labels_path = os.path.join('..', 'data', 'labeled.csv')
labeled_images_pkl_out = os.path.join(main_data_dir, 'labeled_info.pkl')
labeled_images_df_out = os.path.join(main_data_dir, 'labeled_info.csv')


def generate_scored_images_info_file(min_score, background_threshold=None):
    """
    Generates a cluster information file for the scored images.
    @param min_score: only consider images above given score threshold
    @type min_score: float, or None (no threshold if None)
    @param background_threshold:
    @type background_threshold:
    @return: No return, saves the scored image cluster information both as a .pkl and a .csv
    """
    scored_image_ids = [x.replace('.png', '') for x in os.listdir(scored_images_dir)]
    scores_df = pd.read_csv(scores_path, index_col='Id')
    scores_df.index = scores_df.index.astype(str)

    if min_score is not None:
        scored_image_ids = scores_df[scores_df.Actual >= min_score].index.values

    image_data = {}
    for ind, image_id in enumerate(tqdm(scored_image_ids)):
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
    Generates a cluster information file for the labeled images, stores it in `../data/labeled_info.pkl`
    Currently it only considers real images.
    """
    labels_df = pd.read_csv(labels_path, index_col='Id')
    labels_df.index = labels_df.index.astype(str)
    labels_df = labels_df[labels_df.Actual == 1.0]

    real_image_ids = labels_df.index.values
    image_data = {}
    for ind, image_id in enumerate(tqdm(real_image_ids)):
        image_path = os.path.join(labeled_images_dir, "{}.png".format(image_id))
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        data = extract_image_information(img, background_threshold=background_threshold)
        image_data[image_id] = data

    print("Saving pkl to: {}".format(labeled_images_pkl_out))
    with open(labeled_images_pkl_out, 'wb') as f:
        pickle.dump(image_data, f)


def extract_all_information_query(images_dir):
    """
    Generates a cluster information file for the query images. Used only for extracting features for simple cluster-based
    feature regressor.
    @param images_dir: path to directory containing query images
    @type images_dir: str
    @return: a dictionary of dictionaries of properties {img_id: {property1: value, .., property_n: value}}
    @rtype: dict
    """
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


def update_df(df):
    scored_image_ids = [x.replace('.png', '') for x in os.listdir(scored_images_dir)]
    # df['num_zero_pixels'] = 0
    for ind, image_id in enumerate(tqdm(scored_image_ids)):
        image_path = os.path.join(scored_images_dir, "{}.png".format(image_id))
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        df.at[int(image_id), 'num_zero_pixels'] = num_background_pixels(img)
        print(df.at[int(image_id), 'num_zero_pixels'])

    df.num_zero_pixels = df.num_zero_pixels.astype(int)
    df.to_csv(scored_images_df_out, index_label='id')


def load_scored_info(format='df'):
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
    """
    Converts the data dict for query images to pd.DataFrame
    """
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
    """
    Converts the data dict for query images to pd.DataFrame
    """
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
    parser = argparse.ArgumentParser(description='Creates a file containing galaxy information about the labeled or scored images.')
    parser.add_argument('--dataset', type=str, choices=['scored', 'labeled'], default='scored', help='scored or labeled dataset')
    parser.add_argument('--background_threshold', type=int, default=5, help='minimum pixel intensity (0-255) for a '
                                                                            'pixel to be considered as part of a '
                                                                            'galaxy')
    parser.add_argument('--min_score', type=float, default=None, help='disregard fake images i.e images with score < min_score,'
                                                                      'used only if dataset is scored. Has no effect on'
                                                                      'labeled images.')
    parser.add_argument('--update_only', action='store_true', help='If true, will only update the old scored information'
                                                                   'file with new cluster properties.')

    args = parser.parse_args()

    if args.update_only:
        df = load_scored_info(format='df')
        update_df(df)
        sys.exit()

    if args.dataset == 'scored':
        generate_scored_images_info_file(args.min_score, background_threshold=args.background_threshold)
    else:
        generate_labeled_images_info_file(background_threshold=args.background_threshold)