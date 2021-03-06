import sys
import os
import argparse
import matplotlib.pyplot as plt
from generators.baseline_1 import BaselineGenerativeModel
from generators.two_stage_sampling_gan_model import TwoStageModel, TwoStageConditionalModel
from generators.two_stage_sampling_simple_model import TwoStageSimpleModel
from generators.baseline_2 import SmallLargeClustersGenerativeModel
from generators.two_stage_combined_model import TwoStageCombinedModel
from regressors.RESNET import ResnetRegressor
from regressors.random_forest import RandomForestRegressor
from generators.full_galaxy_gan_generator import FullGalaxyGan
from PIL import Image
import numpy as np

class DummyRegressor:
    def score(self, images):
        return [0 for _ in range(len(images))]


def get_generator(args):
    if args.generator == 'baseline':
        model = BaselineGenerativeModel(mean_num_galaxies=20,
                                        std_num_galaxies=10,
                                        mean_galaxy_size=50,
                                        std_galaxy_size=40,
                                        image_width=1000,
                                        image_height=1000)
    elif args.generator == 'baseline_small_large':
        model = SmallLargeClustersGenerativeModel(1000, 1000, use_visual_heuristic=False)
    elif args.generator == 'baseline_small_large_with_heuristic':
        model = SmallLargeClustersGenerativeModel(1000, 1000, use_visual_heuristic=True)
    elif args.generator == 'two_stage_gan':
        model = TwoStageModel(image_width=1000,
                              image_height=1000)
    elif args.generator == 'two_stage_baseline':
        model = TwoStageSimpleModel(image_height=1000,
                                    image_width=1000,
                                    mean_num_galaxies=args.mean_num_galaxies)
    elif args.generator == 'two_stage_conditional':
        model = TwoStageConditionalModel(image_width=1000,
                              image_height=1000)
    elif args.generator == 'combined':
        model = TwoStageCombinedModel()
    elif args.generator == 'full_galaxy_gan':
        model = FullGalaxyGan()
    else:
        raise Exception("model does not exist")
    
    return model


def get_regressor(args):
    if args.regressor == 'resnet':
        return ResnetRegressor()
    elif args.regressor == 'rf':
        return RandomForestRegressor()
    elif args.regressor == 'dummy':
        return DummyRegressor()
    else:
        raise Exception("model does not exist")

def save_images(images, scores, threshold, name):
    indices_to_save = scores >= threshold
    scores_to_save = scores[indices_to_save]
    images_to_save = [images[i] for i in range(len(images)) if indices_to_save[i]]
    # images_to_save = [img.reshape(1, 1000, 1000) for img in images_to_save]
    save_dir = "../data/saved_images_" + name
    if os.path.exists(save_dir):
        print("WARNING: directory '{}' already exists. Please delete if you want to overwrite.".format(save_dir))
        return
    os.mkdir(save_dir)
    for index, image in enumerate(images_to_save):
        Image.fromarray(image.astype(np.uint8)).save(os.path.join(save_dir, str(scores_to_save[index])[:3] + "_"+str(index))+".png", "PNG")


def evaluate(generator, regressor, n_images, visualize=False, save_score_threshold=-1, name=""):
    images = generator.generate(n_images)
    scores = np.asarray(regressor.score(images))
    if (save_score_threshold != -1):
        save_images(images, scores, save_score_threshold, name)
    if visualize:
        assert n_images >= 2

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        ax1.imshow(images[0], cmap='gray', vmin=0, vmax=255)
        ax1.set_title('Score: {}'.format(scores[0]))

        ax2.imshow(images[1], cmap='gray', vmin=0, vmax=255)
        ax2.set_title('Score: {}'.format(scores[1]))

        ax3.hist(scores)
        ax3.set_title("Histogram of scores")
        
        plt.figtext(0.01, 0.01, 'Mean: ' + str(scores.mean()), horizontalalignment='left')
        plt.figtext(0.7, 0.01, 'StdDev: ' + str(scores.std()), horizontalalignment='left')
        plt.show()

    print("Mean score: ", scores.mean())
    print("Stdev: ", scores.std())
    print("Top 10 scores: ", scores[np.argsort(scores)][-10:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates a given generator model using a regressor model.')
    # general script arguments
    parser.add_argument('--generator', type=str, choices=['baseline',
                                                          'two_stage_gan',
                                                          'two_stage_baseline',
                                                          'baseline_small_large',
                                                          'baseline_small_large_with_heuristic',
                                                          'combined',
                                                          'two_stage_conditional',
                                                          'full_galaxy_gan'], default='baseline', help='name of the generator to evaluate')
    parser.add_argument('--regressor', type=str, choices=['resnet', 'rf', 'dummy'], default='resnet', help='name of the regressor that produces the scores for the generator')
    parser.add_argument('--n_images', type=int, default=16, help='number of images to evaluate on')
    parser.add_argument('--save_images', type=int, default=-1, help='save images with score greater than this, -1 dont save')
    parser.add_argument('--visualize', action='store_true', help='if enabled displays images along with their score')

    # baseline models only arguments
    parser.add_argument('--mean_num_galaxies', type=int, default=20, help='Mean of normal distribution of the number '
                                                                          'of galaxies. Used in baseline position '
                                                                          'generator')

    args = parser.parse_args()

    generator = get_generator(args)
    regressor = get_regressor(args)

    evaluate(generator, regressor, args.n_images, visualize=args.visualize, save_score_threshold=args.save_images, name=args.generator)
