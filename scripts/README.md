This directory contains some useful scripts.

galaxy_patches.py: Script that retrieves small images of just galaxies from the full images.

example run: python galaxy_patches.py --min_score 2

evaluate_generator.py: Script that takes a generation method as well as a scoring method and evaluates the former using the latter.

example run: python evaluate_generator.py --n_images 1000 --visualize --generator two_stage_gan --regressor rf

create_images: Script use to generate the figures used in the paper.