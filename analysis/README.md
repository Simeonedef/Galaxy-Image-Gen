This directory contains files used to preform analysis on the data, and experiments on our models. Each of the scripts has a "-h" option for details.

### Initial data analysis
*generate_cluster_information_file.py*:
Generates a csv file with information about the clusters present in each image (such as number of clusters, size and intensity for each cluster).
Precomputed file can be found in _____.  To create the file from scratch, run *python generate_cluster_information_file.py*, followed by
*python generate_cluster_information_file.py --update_only*

*threshold_helper.py*:
Visualization tool we used to understand the effect of different intensity threshold values in determining which clusters in an image are detected as galaxies.

*visualize_clusters.py*:
Visualization tool we used at the start of the project to visually understand the the almost invisible faint clusters in the images.

*real_image_statistics.ipynb*:
A notebook where we perform data analysis on the extracted cluster properties.
Depends on data/scored_info.csv

### Experiments on generative models
*analysis_patches_and_positions.ipynb*:
Notebook containing two experiments on the two-stage generative model: one to analyse how the position of the
patches influences the score and one to analyse how the choice of patches
influences the score. 

### Experiments on regressors
FFT visualize: 
Visualization tool for Fourier transform representation of images with 
different scores. 
