This directory contains jupyter notebooks used to train the various methods we used.

#### Generative models
Both of these models require datasets to be generated using two scripts. The first one generates galaxy patches, and is not included in the submission. In the scripts directory, run `python galaxy_patches.py`. 

The files for the second one *are already included. If necessary*, in the analysis directory, run `python generate_cluster_information_file.py --dataset labeled --background_threshold 5 --min_score 2` followed by `python generate_cluster_information_file.py --dataset scored --background_threshold 5 --min_score 2`. These both take some time.

* *two_stage_galaxy_sampling*: Contains both conditional and unconditional gans used for the two stage generation method, which is one of our main contributions.

* *full_image_generation*: Contains a VAE and different GANs used to attempt to directly generate full images.

Note: The baselines require no training, hence there's no file here with them.

#### Regressor models 
* *galaxy_features_regressor*: Regressor baseline #1, uses manually extracted features based on clusters.
    - *feature_selection.ipynb*: extracts (and analyzes) cluster-based features, to be later used as features for different
     regression models.\
     Prerequisite: */data/scored_info.csv*
     Produces: *train_X.csv*, *train_y.csv*, *test_X.csv*. These files are already precomputed and uploaded the directory,
     it's not necessary to run the notebook from scratch. 
   - *baseline_reg.ipynb*: evaluates multiple models trained on the previously extracted cluster-based features.
   - *xgb_hyperparam_tuning.ipynb*: tunes the best performing model (xgboost). 
   
* *cnn_reg*: Regressor baseline #2, uses a CNN to directly compute the score. *CNNRegressor.ipynb* trains a CNN from scratch,
*RESNET.ipynb* uses a pre-trained backbone network.

* *main_regressor*: Contains the best performing XGBoost regressor we trained as well as some feature extraction and exploration notebooks.
