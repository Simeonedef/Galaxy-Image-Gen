import sys
import os
# carlos is too lazy to add the root to his path, so we have to do this
# TAs if you're seeing this I will gladly give up 0.5 of our 'implementation' grade for this
sys.path.append(os.path.abspath('..'))
import argparse
import matplotlib.pyplot as plt
from generators.simple_generative_model import BaselineGenerativeModel
from regressors.RESNET import ResnetRegressor


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
        return model
    else:
        raise Exception("model does not exist")


def get_regressor(args):
    if args.regressor == 'resnet':
        return ResnetRegressor()
    elif args.regressor == 'dummy':
        return DummyRegressor()
    else:
        raise Exception("model does not exist")


def evaluate(generator, regressor, n_images, visualize=False):
    images = generator.generate(n_images)
    scores = regressor.score(images)

    if visualize:
        assert n_images >= 2

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.imshow(images[0], cmap='gray')
        ax1.set_title('Score: {}'.format(scores[0]))

        ax2.imshow(images[1], cmap='gray')
        ax2.set_title('Score: {}'.format(scores[1]))

        plt.show()

    print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates a given generator model using a regressor model.')
    parser.add_argument('--generator', type=str, choices=['baseline'], default='baseline', help='name of the generator to evaluate')
    parser.add_argument('--regressor', type=str, choices=['resnet, dummy'], default='resnet', help='name of the regressor that produces the scores for the generator')
    parser.add_argument('--n_images', type=int, default=16, help='number of images to evaluate on')
    parser.add_argument('--visualize', action='store_true', help='if enabled displays images along with their score')
    args = parser.parse_args()

    generator = get_generator(args)
    regressor = get_regressor(args)

    evaluate(generator, regressor, args.n_images, visualize=args.visualize)
