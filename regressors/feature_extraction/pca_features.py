from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
from PIL import Image
pca_image_size = (128, 128)
data_dir = "../data/"

scored_dir = os.path.join(data_dir, "scored")
query_dir = os.path.join(data_dir, "query")

n_components=0.9 #means it will return the Eigenvectors that have the 90% of the variation in the dataset

scored_pca = PCA(n_components=n_components)
scored_images = []
image_names = []
for image_file in os.listdir(scored_dir):
    image = Image.open(os.path.join(scored_dir, image_file)).convert("L").resize(pca_image_size)
    image = np.asarray(image)/255
    scored_images.append(image.flatten())
    image_names.append(image_file[:-4])
scored_images = np.asarray(scored_images)
features = scored_pca.fit_transform(scored_images)

print(features.shape)
features = pd.DataFrame(features)
features['Id'] = np.asarray(image_names)
features = features.set_index("Id")
features.to_csv(os.path.join(data_dir, "scored_features.csv"))

query_images = []
image_names = []
for image_file in os.listdir(query_dir):
    image = Image.open(os.path.join(query_dir, image_file)).convert("L").resize(pca_image_size)
    image = np.asarray(image)/255
    query_images.append(image.flatten())
    image_names.append(image_file[:-4])

query_images = np.asarray(query_images)
features = scored_pca.transform(query_images)

features = pd.DataFrame(features)
features['Id'] = np.asarray(image_names)
features = features.set_index("Id")
features.to_csv(os.path.join(data_dir, "query_features.csv"))