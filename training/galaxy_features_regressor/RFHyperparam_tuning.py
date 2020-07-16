import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

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

