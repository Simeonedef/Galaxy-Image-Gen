# Galaxy-Image-Gen

*TODO: Insert high level description here (after we finish report)*

### Setup

Place the 3 image folders *labeled*, *query* and *scored* in the data directory.

Run `python setup.py develop`.

Run `conda env create -f environment.yml`, followed by `conda activate galaxy_gen`.

Under the scripts directory, run `python galaxy_patches.py`.

### Project directories
1. *generators*: contains the generative model classes for all our generative models
2. *regressors*: contains the regressor model classes for all our regression models
3. *training*: contains the notebooks and scripts used to train our generative/regressive models
4. *common*: common utils used throughout the project
5. *analysis*: experiments on our models and data analysis
6. *scripts*: self-explanatory

Each directory has its own README explaining how to run its scripts/notebooks.
