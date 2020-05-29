import sys
import os
import argparse
import matplotlib.pyplot as plt
from generators.simple_generative_model import BaselineGenerativeModel
from generators.two_stage_sampling_gan_model import TwoStageModel
from generators.two_stage_sampling_simple_model import TwoStageSimpleModel
from regressors.RESNET import ResnetRegressor
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
    elif args.generator == 'two_stage_gan':
        model = TwoStageModel(image_width=1000,
                              image_height=1000)
    elif args.generator == 'two_stage_baseline':
        model = TwoStageSimpleModel(image_height=1000,
                                    image_width=1000,
                                    mean_num_galaxies=args.mean_num_galaxies)
    else:
        raise Exception("model does not exist")
    
    return model


def get_regressor(args):
    if args.regressor == 'resnet':
        return ResnetRegressor()
    elif args.regressor == 'dummy':
        return DummyRegressor()
    else:
        raise Exception("model does not exist")


def evaluate(generator, regressor, n_images, visualize=False):
    images = generator.generate(n_images)
    scores = np.asarray(regressor.score(images))
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates a given generator model using a regressor model.')
    # general script arguments
    parser.add_argument('--generator', type=str, choices=['baseline',
                                                          'two_stage_gan',
                                                          'two_stage_baseline'], default='baseline', help='name of the generator to evaluate')
    parser.add_argument('--regressor', type=str, choices=['resnet', 'dummy'], default='resnet', help='name of the regressor that produces the scores for the generator')
    parser.add_argument('--n_images', type=int, default=16, help='number of images to evaluate on')
    parser.add_argument('--visualize', action='store_true', help='if enabled displays images along with their score')

    # baseline models only arguments
    parser.add_argument('--mean_num_galaxies', type=int, default=20, help='Mean of normal distribution of the number '
                                                                          'of galaxies. Used in baseline position '
                                                                          'generator')

    args = parser.parse_args()

    generator = get_generator(args)
    regressor = get_regressor(args)

    evaluate(generator, regressor, args.n_images, visualize=args.visualize)
