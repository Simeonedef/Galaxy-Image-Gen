import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, make_scorer

if __name__ == "__main__":
    train_X = pd.read_csv('train_X.csv', dtype='float')
    train_y = pd.read_csv('train_y.csv', dtype='float')

    test_X_df = pd.read_csv('test_X.csv', dtype='float')

    assert train_X.shape[1] == test_X_df.shape[1]

    na_indices = train_X[train_X.isna().any(axis=1)].index.values
    train_X.drop(na_indices, inplace=True)
    train_y.drop(na_indices, inplace=True)

    print(train_X.shape)
    print(train_X.columns.values)

    X = train_X.to_numpy()
    y = train_y.to_numpy()
    test_X = test_X_df.to_numpy()

    # hyperparams = {
    #     'n_estimators': [1000, 2000, 5000],
    #     "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    #      "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    #      "min_child_weight": [1, 3, 5, 7],
    #      "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    #      "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
    # }
    hyperparams = {
        "clf__objective": ["reg:squarederror"],
        "clf__learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "clf__n_estimators": [1000, 2000, 5000],
        "clf__max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        "clf__gamma": [0.1, 0.2, 0.3, 0.4, 0.5],
        "clf__subsample": [0.7],
        "clf__colsample_bytree": [0.3, 0.4, 0.5, 0.7, 1.0]
    }

    pipeline = Pipeline([('transformer', StandardScaler()),
                         ('clf', XGBRegressor())])

    search = GridSearchCV(pipeline, hyperparams, cv=5, verbose=2, n_jobs=-1, refit=True, scoring=make_scorer(mean_absolute_error))
    search.fit(X, np.ravel(y))
    print(search.__dict__)
    print('======================================')
    model = search.best_estimator_
    print("Best estimator: ", model)
    print('best score: ', search.best_score_)

    with open('xgboost_gridsearch.pickle', 'wb') as f:
        pickle.dump(search.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    predictions = model.predict(test_X)
    predictions = np.clip(predictions, 0, 8)

    print("Testing predictions: ", predictions)

    predictions_file_ids = [x.replace('.png', '') for x in os.listdir(os.path.join('../..', 'data', 'query'))]
    results = {'Id': predictions_file_ids, 'Predicted': predictions.reshape(-1)}
    results = pd.DataFrame(data=results, dtype='float')
    results.Id = results.Id.astype('int')
    results.set_index('Id', inplace=True)

    results.to_csv('out.csv', index=True)


