import argparse
import torch
import pandas as pd
import os
import numpy as np
from PIL import Image
import pickle
from scipy import ndimage
from xgboost import XGBRegressor
window_len = 5
model_file = "../regressors/random_forest.pkl"

class RandomForestRegressor:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        print("Loading model...")
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)
        print("Model loaded!")
    
    def feature_extraction(self, images):
        image_histograms = []
        image_fft = []
        for image in images:
            image_histograms.append(Image.fromarray(image).histogram())
            image = image.astype(np.float32)
            f = np.fft.fft2(image)
            fshift = np.fft.fftshift(f)
            psd1d = self.GetPSD1D(np.abs(fshift) ** 2)
            # img_features = []
            # for i in range(len(psd1d) - window_len + 1):
            #     element = 0
            #     for j in range(window_len):
            #         element = element + psd1d[i + j]
            #     img_features.append(element / window_len)
            # image_fft.append(np.asarray(img_features))
            windowed_average = np.convolve(psd1d, np.ones((window_len,))/window_len, mode='valid')
            image_fft.append(windowed_average)
        return np.hstack((np.asarray(image_histograms), np.asarray(image_fft)))

    def GetPSD1D(self, psd2D):
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

    def score(self, images):
        print("Scoring images:")
        features = self.feature_extraction(images)
        predictions = self.model.predict(features)
        return predictions