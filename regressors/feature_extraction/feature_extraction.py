import numpy as np
import pandas as pd
import os
from PIL import Image
from skimage.feature import hog, local_binary_pattern
import argparse
import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import ndimage
data_dir = "../../data/"

scored_dir = os.path.join(data_dir, "scored")
query_dir = os.path.join(data_dir, "query")

scores_file = os.path.join(data_dir, "scored.csv")

def extract(dir, hog='n', lbp='n', fft='y', output_file=""):
    scores = pd.read_csv(scores_file, index_col="Id")
    image_dir = os.path.join(data_dir, dir)
    image_histograms = []
    hog_histograms = []
    lbp_histograms = []
    fft_histograms = []
    image_scores = []
    image_names = []

    for image_file in tqdm.tqdm(os.listdir(image_dir)):
        image = Image.open(os.path.join(image_dir, image_file)).convert("L")
        image_histograms.append(np.asarray(image.histogram()))
        image_names.append(image_file[:-4])
        image = np.asarray(image, dtype=np.float32)
        if hog == 'y':
            hog_histograms.append(get_hog_histogram(image))
        if lbp == 'y':
            lbp_histograms.append(get_lbp_histogram(image))
        if fft == 'y':
            fft_histograms.append(get_fft(image))
        if dir == "query":
            image_scores.append(-1)
        else:
            image_scores.append(scores.loc[int(image_file[:-4])].Actual)
    
    output = None
    image_histograms = np.asarray(image_histograms)
    hog_histograms = np.asarray(hog_histograms)
    lbp_histograms = np.asarray(lbp_histograms)
    fft_histograms = np.asarray(fft_histograms)
    image_scores = np.asarray(image_scores)
    output = pd.DataFrame(image_histograms)
    if hog == 'y':
        hog_features = pd.DataFrame(hog_histograms)
        output = pd.concat([output, hog_features], ignore_index=True, axis=1)
    if lbp == 'y':
        lbp_features = pd.DataFrame(lbp_histograms)
        output = pd.concat([output, lbp_features], ignore_index=True, axis=1)
    if fft == 'y':
        fft_features = pd.DataFrame(fft_histograms)
        output = pd.concat([output, fft_features], ignore_index=True, axis=1)
    output['Id'] = np.asarray(image_names)
    output['Actual'] = image_scores
    output = output.set_index("Id")
    output.to_csv(os.path.join(data_dir, "{}_features_{}.csv".format(dir, output_file)))


def get_fft(image, window_len=5):
    image = np.asarray(image, dtype=np.float32)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    psd1d = GetPSD1D(np.abs(fshift) ** 2)
    windowed_average = np.convolve(psd1d, np.ones((window_len,))/window_len, mode='valid')
    # img_features = []
    # for i in range(len(psd1d) - window_len + 1):
    #     element = 0
    #     for j in range(window_len):
    #         element = element + psd1d[i + j]
    #     img_features.append(element / window_len)
    # image_fft.append(np.asarray(img_features))
    return windowed_average

def GetPSD1D(psd2D):
        h  = psd2D.shape[0]
        w  = psd2D.shape[1]
        wc = w//2
        hc = h//2

        # create an array of integer radial distances from the center
        Y, X = np.ogrid[0:h, 0:w]
        r    = np.hypot(X - wc, Y - hc).astype(np.int)

        # SUM all psd2D pixels with label 'r' for 0<=r<=wc
        # NOTE: this will miss power contributions in 'corners' r>wc
        psd1D = ndimage.sum(psd2D, r, index=np.arange(0, wc))

        return psd1D

def get_lbp_histogram(image, num_points=24, radius=8, method="uniform"):
    lbp = local_binary_pattern(image, num_points, radius, method=method)
    (hist, _) = np.histogram(lbp.ravel(),
    		bins=np.arange(0, num_points + 3),
    		range=(0, num_points + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-15)
    return hist

def get_hog_histogram(image, orientations=9, pixels_per_cell=(32, 32), visualize=False, feature_vector=True):
    return hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, visualize=visualize, feature_vector=feature_vector)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates a file containing extracted features from images.')
    parser.add_argument('--hog', type=str, choices=['y', 'n'], default='n', help='include hog features')
    parser.add_argument('--lbp', type=str, choices=['y', 'n'], default='n', help='include lbp features')
    parser.add_argument('--fft', type=str, choices=['y', 'n'], default='y', help='include fft features')
    parser.add_argument('--output', type=str, default='', help='output file name')

    args = parser.parse_args()
    extract('scored', hog=args.hog, lbp=args.lbp, fft=args.fft, output_file=args.output)
    extract('query', hog=args.hog, lbp=args.lbp, fft=args.fft, output_file=args.output)