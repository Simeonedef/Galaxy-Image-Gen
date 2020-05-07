import pickle
from analysis.generate_cluster_information_file import image_information_file_path


def load_image_data(file_path):
    with open(image_information_file_path, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == "__main__":
    image_data = load_image_data(image_information_file_path)
    print(image_data)
