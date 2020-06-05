import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from generators.simple_generative_model import draw_galaxy


def geom(peak, p, size):
    return stats.geom.rvs(p, size=size).astype(int) + peak - 1


class SmallLargeClustersGenerativeModel:
    """
    A simple generative model. It treats small galaxies and large galaxies separately.
    - The number of small/large galaxies is sampled from two separate normal distributions
    - The center of each galaxy is sampled from a uniform distribution over the image pixels
    - The size of each galaxy is sampled from a geometric distribution (separate for large/small)
    - The intensity of the galaxy sampled from a geometric distribution. The center pixel is the brightest, and the brightness
      goes down as we move away from the center pixel.
    - The shape of the galaxy is always:
                          *
                         ***
                        *****
                         ***
                          *
    """

    def __init__(self, image_height, image_width,
                 mean_num_large_galaxies=50, std_num_large_galaxies=15,
                 mean_num_small_galaxies=10, std_num_small_galaxies=5,
                 peak_small_galaxies=1, p_small_galaxies=0.7,
                 peak_large_galaxies=1, p_large_galaxies=0.3):
        self.p_large_galaxies = p_large_galaxies
        self.peak_large_galaxies = peak_large_galaxies
        self.p_small_galaxies = p_small_galaxies
        self.peak_small_galaxies = peak_small_galaxies
        self.mean_num_small_galaxies = mean_num_small_galaxies
        self.std_num_small_galaxies = std_num_small_galaxies
        self.std_num_large_galaxies = std_num_large_galaxies
        self.mean_num_large_galaxies = mean_num_large_galaxies

        self.image_width = image_width
        self.image_height = image_height

    def sample(self):
        self.num_large_galaxies, self.num_small_galaxies = self.sample_num_galaxies()
        self.galaxy_centers_large, self.galaxy_centers_small = self.sample_galaxy_centers()
        self.galaxy_sizes_large, self.galaxy_sizes_small = self.sample_galaxy_sizes()
        self.intensities_large, self.intensities_small = self.sample_galaxy_intensities()

    def generate(self, n_images):
        print("Generating images:")
        return [self.draw() for _ in tqdm(range(n_images))]

    def draw(self, show=False, draw_large=True):
        self.sample()
        assert self.num_large_galaxies is not None

        img = np.ones((self.image_height, self.image_width))

        if draw_large:
            for center, size, intensity in zip(self.galaxy_centers_large, self.galaxy_sizes_large, self.intensities_large):
                draw_galaxy(img, center, size, intensity)
        for center, size, intensity in zip(self.galaxy_centers_small, self.galaxy_sizes_small, self.intensities_small):
            draw_galaxy(img, center, size, intensity)

        if show:
            plt.imshow(img, cmap='gray')
            plt.show()

        return img

    def sample_num_galaxies(self):
        return max(0, int(np.random.normal(self.mean_num_large_galaxies, self.std_num_large_galaxies))), \
               max(0, int(np.random.normal(self.mean_num_small_galaxies, self.std_num_small_galaxies)))

    def sample_galaxy_centers(self):
        galaxy_centers_large_x = np.random.randint(0, self.image_height, size=self.num_large_galaxies)
        galaxy_centers_large_y = np.random.randint(0, self.image_width, size=self.num_large_galaxies)
        galaxy_centers_small_x = np.random.randint(0, self.image_height, size=self.num_small_galaxies)
        galaxy_centers_small_y = np.random.randint(0, self.image_width, size=self.num_small_galaxies)
        return zip(galaxy_centers_large_x, galaxy_centers_large_y), zip(galaxy_centers_small_x, galaxy_centers_small_y)

    def sample_galaxy_sizes(self):
        return geom(self.peak_large_galaxies, self.p_large_galaxies, size=self.num_large_galaxies), \
               geom(self.peak_small_galaxies, self.p_small_galaxies, size=self.num_small_galaxies)

    def sample_galaxy_intensities(self):
        return geom(1, 0.2, size=self.num_large_galaxies), \
               geom(1, 0.8, size=self.num_small_galaxies)


if __name__ == "__main__":
    model = SmallLargeClustersGenerativeModel(1000, 1000)

    model.draw(show=True)
