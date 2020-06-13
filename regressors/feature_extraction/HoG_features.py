from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
resized_image_size = (128, 128)
data_dir = "../data/"

scored_dir = os.path.join(data_dir, "scored")
query_dir = os.path.join(data_dir, "query")


def get_lbp_histogram(image, num_points=24, radius=8, method="uniform"):
    lbp = local_binary_pattern(image, num_points, radius, method=method)
    (hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, num_points + 3),
			range=(0, num_points + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def get_hog_histogram(image):
    return hog(image, orientations=9, pixels_per_cell=(32, 32), visualize=False, feature_vector=True)

if __name__ == "__main__":
    scored_histograms = []
    lbp_histograms = []
    image_names = []
    for image_file in os.listdir(scored_dir):
        image = Image.open(os.path.join(scored_dir, image_file)).convert("L").resize(resized_image_size)
        image = np.asarray(image)/255.
        # pixels per cell, orientations and cells per block are design choices
        # can also try smaller pixels per cell and ROI
        fd = hog(image, orientations=9, pixels_per_cell=(32, 32), visualize=False, feature_vector=True)
        scored_histograms.append(fd)
        lbp_histograms.append(get_lbp_histogram(image))
        image_names.append(image_file[:-4])
    scored_histograms = np.asarray(scored_histograms)

    hog_features = pd.DataFrame(scored_histograms)
    hog_features.columns = [str(col)+"_hist" for col in hog_features.columns]
    lbp_features = pd.DataFrame(lbp_histograms)
    lbp_features.columns = [str(col)+"_hist" for col in lbp_features.columns]
    features = pd.concat([hog_features, lbp_features], ignore_index=True, axis=1)
    features['Id'] = np.asarray(image_names)

    features = features.set_index("Id")
    features.to_csv(os.path.join(data_dir, "scored_hog_lbp.csv"))

    query_histograms = []
    lbp_histograms = []
    image_names = []
    for image_file in os.listdir(query_dir):
        image = Image.open(os.path.join(query_dir, image_file)).convert("L").resize(resized_image_size)
        image = np.asarray(image)/255.
        # pixels per cell, orientations and cells per block are design choices
        # can also try smaller pixels per cell and ROI
        fd = hog(image, orientations=9, pixels_per_cell=(32, 32), visualize=False, feature_vector=True)
        query_histograms.append(fd)
        lbp_histograms.append(get_lbp_histogram(image))
        image_names.append(image_file[:-4])
    query_histograms = np.asarray(query_histograms)

    hog_features = pd.DataFrame(query_histograms)
    hog_features.columns = [str(col)+"_hist" for col in hog_features.columns]
    lbp_features = pd.DataFrame(lbp_histograms)
    lbp_features.columns = [str(col)+"_hist" for col in lbp_features.columns]
    features = pd.concat([hog_features, lbp_features], ignore_index=True, axis=1)
    features['Id'] = np.asarray(image_names)

    features = features.set_index("Id")
    features.to_csv(os.path.join(data_dir, "query_hog_lbp.csv"))