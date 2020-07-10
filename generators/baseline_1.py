import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from common.galaxy_drawing import draw_galaxy, get_radius
from tqdm import tqdm


class BaselineGenerativeModel:
    """
    A simple generative model.
    - The number of galaxies is sampled from a normal distribution N(?, ?)
    - The center of each galaxy is sampled from a uniform distribution over the image pixels
    - The size of each galaxy is sampled from a normal distribution N(?, ?)
    - The shape of the galaxy is always:
                          *
                         ***
                        *****
                         ***
                          *
    - The intensity of the galaxy is proportional to its size. The center pixel is the brightest, and the brightness
      goes down as we move away from the center pixel.
    """
    def __init__(self, mean_num_galaxies=10, std_num_galaxies=2,
                       mean_galaxy_size=15, std_galaxy_size=6,
                       image_width=1000, image_height=1000):
        self.image_width = image_width
        self.image_height = image_height
        self.mean_num_galaxies = mean_num_galaxies
        self.std_num_galaxies = std_num_galaxies
        self.mean_galaxy_size = mean_galaxy_size
        self.std_galaxy_size = std_galaxy_size

    def sample(self):
        self.num_galaxies = self.sample_num_galaxies()
        self.galaxy_centers = self.sample_galaxy_centers()
        self.galaxy_sizes = self.sample_galaxy_sizes()

    def generate(self, n_images):
        print("Generating images:")
        return [self.draw() for _ in tqdm(range(n_images))]

    def draw(self, show=False):
        self.sample()
        assert self.num_galaxies is not None

        img = np.zeros((self.image_height, self.image_width))
        for center, size in zip(self.galaxy_centers, self.galaxy_sizes):
            radius = get_radius(size)
            print(radius, size)
            draw_galaxy(img, center, get_radius(size), 255, fade=5)

        if show:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            plt.show()

        return img

    def sample_num_galaxies(self):
        return max(0, int(np.random.normal(self.mean_num_galaxies, self.std_num_galaxies)))

    def sample_galaxy_centers(self):
        galaxy_centers_x = np.random.randint(0, self.image_height, size=self.num_galaxies)
        galaxy_centers_y = np.random.randint(0, self.image_width, size=self.num_galaxies)
        return zip(galaxy_centers_x, galaxy_centers_y)

    def sample_galaxy_sizes(self):
        return np.random.normal(self.mean_galaxy_size, self.std_galaxy_size, size=self.num_galaxies).astype(int)


if __name__ == "__main__":
    model = BaselineGenerativeModel()
    model.draw(show=True)
