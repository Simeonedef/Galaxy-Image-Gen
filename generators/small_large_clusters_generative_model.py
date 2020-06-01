import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats


def geom(peak, p, size):
    return stats.geom.rvs(p, size=size).astype(int) + peak - 1


class SmallLargeClustersGenerativeModel:
    """
    A simple common generative model.
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

    def __init__(self, image_height, image_width,
                 mean_num_large_galaxies=50, std_num_large_galaxies=15,
                 mean_num_small_galaxies=10, std_num_small_galaxies=5,
                 peak_small_galaxies=1, p_small_galaxies=0.7,
                 peak_large_galaxies=1, p_large_galaxies=0.2):
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

    def draw(self, show=False):
        self.sample()
        assert self.num_large_galaxies is not None

        img = np.ones((self.image_height, self.image_width))
        for center, size, intensity in zip(self.galaxy_centers_large, self.galaxy_sizes_large, self.intensities_large):
            self.draw_galaxy(img, center, size, intensity)
        for center, size, intensity in zip(self.galaxy_centers_small, self.galaxy_sizes_small, self.intensities_small):
            self.draw_galaxy(img, center, size, intensity)

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


    def draw_galaxy(self, img, center, size, intensity):
        # dummy version, the intensity of the center pixel is literally equal to the galaxy size
        # intensity = size
        fade = 1
        # draw upper half, including center row
        for x_delta in range(size):
            x = center[0] - x_delta
            if x < 0:
                break

            # center and to the left
            for y_delta in range(size - x_delta):
                y = center[1] - y_delta
                if y < 0 or y >= img.shape[1]:
                    break
                img[x][y] = max(intensity - fade * y_delta - fade * x_delta, 0)

            # to the right
            for y_delta in range(size - x_delta):
                y = center[1] + y_delta
                if y < 0 or y >= img.shape[1]:
                    break

                img[x][y] = max(intensity - fade * y_delta - fade * x_delta, 0)

        # draw lower half
        for x_delta in range(1, size):
            x = center[0] + x_delta
            if x >= img.shape[0]:
                break

            # center and to the left
            for y_delta in range(size - x_delta):
                y = center[1] - y_delta
                if y < 0 or y >= img.shape[1]:
                    break
                img[x][y] = max(intensity - fade * y_delta - fade * x_delta, 0)

            # to the right
            for y_delta in range(size - x_delta):
                y = center[1] + y_delta
                if y < 0 or y >= img.shape[1]:
                    break

                img[x][y] = max(intensity - fade * y_delta - fade * x_delta, 0)


if __name__ == "__main__":
    model = SmallLargeClustersGenerativeModel(1000, 1000)

    model.draw(show=True)
