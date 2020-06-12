from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
pca_image_size = (128, 128)
data_dir = "../data/"

scored_dir = os.path.join(data_dir, "scored")
query_dir = os.path.join(data_dir, "query")

scored_histograms = []
image_names = []
for image_file in os.listdir(scored_dir):
    image = Image.open(os.path.join(scored_dir, image_file)).convert("L").resize(pca_image_size)
    image = np.asarray(image)/255.
    # pixels per cell, orientations and cells per block are design choices
    # can also try smaller pixels per cell and ROI
    fd = hog(image, orientations=9, pixels_per_cell=(32, 32), visualize=False, feature_vector=True)
    
    scored_histograms.append(fd)
    image_names.append(image_file[:-4])
scored_histograms = np.asarray(scored_histograms)


features = pd.DataFrame(scored_histograms)
features['Id'] = np.asarray(image_names)
features = features.set_index("Id")
features.to_csv(os.path.join(data_dir, "scored_hog.csv"))

query_histograms = []
image_names = []
for image_file in os.listdir(query_dir):
    image = Image.open(os.path.join(query_dir, image_file)).convert("L").resize(pca_image_size)
    image = np.asarray(image)/255.
    # pixels per cell, orientations and cells per block are design choices
    # can also try smaller pixels per cell and ROI
    fd = hog(image, orientations=8, pixels_per_cell=(64, 64), visualize=False, feature_vector=True)
    lbp = local_binary_pattern(image, 24, 8, method="uniform")
    plt.imshow(lbp, cmap="gray")
    print("here")
    input()
    query_histograms.append(fd)
    image_names.append(image_file[:-4])
query_histograms = np.asarray(query_histograms)

features = pd.DataFrame(query_histograms)
features['Id'] = np.asarray(image_names)
features = features.set_index("Id")
features.to_csv(os.path.join(data_dir, "query_hog.csv"))