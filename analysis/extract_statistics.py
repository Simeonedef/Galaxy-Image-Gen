import pickle
from analysis.generate_cluster_information_file import scored_images_pkl_out


def load_image_data(file_path):
    with open(scored_images_pkl_out, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == "__main__":
    image_data = load_image_data(scored_images_pkl_out)
    print(image_data)
