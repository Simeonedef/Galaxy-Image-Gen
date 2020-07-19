This directory contains jupyter notebooks used to train the various methods we used.

* *two_stage_galaxy_sampling*: Contains both conditional and unconditional gans used for the two stage generation method, which is one of our main contributions.

* *full_image_generation*: Contains a VAE and different GANs used to attempt to directly generate full images.

* *main_regressor*: Contains the best performing XGBoost regressor we trained as well as some feature extraction and exploration notebooks.

* *cnn_reg*: 

* *galaxy_features_regressor*: Regressor baseline #1, uses manually extracted features based on clusters.
    - *feature_selection.ipynb*: extracts (and analyzes) cluster-based features, to be later used as features for different
     regression models.\
     Prerequisite: */data/scored_info.csv* (download from https://polybox.ethz.ch/index.php/s/BkYCGbxdSVkzavW) \
     Produces: *train_X.csv*, *train_y.csv*, *test_X.csv*. These files are already precomputed and uploaded the directory,
     it's not necessary to run the notebook from scratch. 
   - *baseline_reg.ipynb*: evaluates multiple models trained on the previously extracted cluster-based features.
   - *xgb_hyperparam_tuning.ipynb*: tunes the best performing model (xgboost). 