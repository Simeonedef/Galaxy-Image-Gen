import ast
import pickle
import json
import itertools
from collections import Counter
from tqdm import tqdm
import pandas as pd
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import swifter

from analysis.generate_cluster_information_file import load, extract_all_information_query, to_df_query
from common.image_processing import pixel_intensity_histogram


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


train_X = pd.read_csv('train_X.csv')
train_y = pd.read_csv('train_y.csv')

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
train_y_scaled = scaler.fit_transform(train_y)
train_X_na = train_X.fillna(0)
train_X_na_scaled = scaler.fit_transform(train_X_na)

"""
    Tunes hyperparameters for RandomForest
"""
hyperparameters = {
    'n_estimators': [100, 500, 1000, 5000],
    'criterion' : ['mse', 'mae'],
    'max_depth': [3, 6, 8, 20, None],
    'min_samples_split' : [2, 4, 8, 16],
    'min_samples_leaf' : [1, 2, 4],
    'max_features' : ['auto', 'sqrt', 'log2']
}

randomforest = RandomForestRegressor(verbose=2)

clfRF = RandomizedSearchCV(randomforest,
                         hyperparameters,
                         random_state=1, n_iter=100, cv=3, verbose=2, n_jobs=-1)
clfRF.fit(train_X_na_scaled, np.ravel(train_y_scaled))
print(clfRF.__dict__)
print('======================================')
print(clfRF.best_estimator_)
print('best score: ', clfRF.best_score_)

with open('clfRf.pickle', 'wb') as f:
    pickle.dump(clfRF.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

# f = open('RF__dict__.txt', 'w+')
# f.write(json.dumps(clfRF.__dict__))
# f.write('=======================================\n')
# f.write(json.dumps(clfRF.best_estimator_))
# f.write('best score: ')
# f.write(json.dumps(clfRF.best_score_))
# f.close()

