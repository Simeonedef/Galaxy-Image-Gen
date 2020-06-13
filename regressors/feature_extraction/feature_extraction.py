import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
import argparse
import tqdm
resized_image_size = (128, 128)
data_dir = "../../data/"

scored_dir = os.path.join(data_dir, "scored")
query_dir = os.path.join(data_dir, "query")

def extract(dir, hog='n', lbp='n', fft='y'):
    image_dir = os.path.join(data_dir, dir)
    hog_histograms = []
    lbp_histograms = []
    fft_histograms = []
    image_names = []
    for image_file in tqdm.tqdm(os.listdir(image_dir)):
        image = Image.open(os.path.join(image_dir, image_file)).convert("L").resize(resized_image_size)
        image = np.asarray(image, dtype=np.float32)
        # pixels per cell, orientations and cells per block are design choices
        # can also try smaller pixels per cell and ROI
        if hog == 'y':
            fd = hog(image, orientations=9, pixels_per_cell=(32, 32), visualize=False, feature_vector=True)
            hog_histograms.append(fd)
        if lbp == 'y':
            lbp_histograms.append(get_lbp_histogram(image))
        if fft == 'y':
            fft_histograms.append(get_fft_histogram(image))
        image_names.append(image_file[:-4])
    
    output = None
    hog_histograms = np.asarray(hog_histograms)
    lbp_histograms = np.asarray(lbp_histograms)
    fft_histograms = np.asarray(fft_histograms)

    if hog == 'y':
        hog_features = pd.DataFrame(hog_histograms)
        if output is None:
            output = hog_features
        else:
            output = pd.concat([output, hog_features], ignore_index=True, axis=1)
    if lbp == 'y':
        lbp_features = pd.DataFrame(lbp_histograms)
        if output is None:
            output = lbp_features
        else:
            output = pd.concat([output, lbp_features], ignore_index=True, axis=1)
    if fft == 'y':
        fft_features = pd.DataFrame(fft_histograms)
        if output is None:
            output = fft_features
        else:
            output = pd.concat([output, fft_features], ignore_index=True, axis=1)
    output['Id'] = np.asarray(image_names)

    output = output.set_index("Id")
    output.to_csv(os.path.join(data_dir, "{}_features.csv".format(dir)))

def get_fft_histogram(image, n_bins=100):
    eps = 1e-7
    fft = np.fft.fft2(image)
    fft_shift = np.abs(np.fft.fftshift(fft))**2
    return np.histogram(fft_shift, bins=n_bins)[0]

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
    parser = argparse.ArgumentParser(description='Creates a file containing extracted features from images.')
    parser.add_argument('--hog', type=str, choices=['y', 'n'], default='n', help='include hog features')
    parser.add_argument('--lbp', type=str, choices=['y', 'n'], default='n', help='include lbp features')
    parser.add_argument('--fft', type=str, choices=['y', 'n'], default='y', help='include fft features')

    args = parser.parse_args()
    extract('scored', hog=args.hog, lbp=args.lbp, fft=args.fft)
    extract('query', hog=args.hog, lbp=args.lbp, fft=args.fft)