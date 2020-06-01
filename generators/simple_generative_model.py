import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class BaselineGenerativeModel:
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
    def __init__(self, mean_num_galaxies=20, std_num_galaxies=2,
                       mean_galaxy_size=15, std_galaxy_size=2,
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
        print ("Generating images:")
        return [self.draw() for _ in tqdm(range(n_images))]

    def draw(self, show=False):
        self.sample()
        assert self.num_galaxies is not None

        img = np.ones((self.image_height, self.image_width))
        for center, size in zip(self.galaxy_centers, self.galaxy_sizes):
            self.draw_galaxy(img, center, size)

        if show:
            plt.imshow(img, cmap='gray')
            plt.show()

        return img

    def sample_num_galaxies(self):
        # normal distribution will sometimes produce negative number of galaxies
        # this is a quick workarround, but another distribution may be more appropriate
        # e.g. something discrete like a binomial or a poisson
        return max(0, int(np.random.normal(self.mean_num_galaxies, self.std_num_galaxies)))

    def sample_galaxy_centers(self):
        galaxy_centers_x = np.random.randint(0, self.image_height, size=self.num_galaxies)
        galaxy_centers_y = np.random.randint(0, self.image_width, size=self.num_galaxies)
        return zip(galaxy_centers_x, galaxy_centers_y)

    def sample_galaxy_sizes(self):
        return np.random.normal(self.mean_galaxy_size, self.std_galaxy_size, size=self.num_galaxies).astype(int)

    def draw_galaxy(self, img, center, size):
        # dummy version, the intensity of the center pixel is literally equal to the galaxy size
        intensity = size
        fade = 3
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
    model = BaselineGenerativeModel(mean_num_galaxies=4, std_num_galaxies=1,
                                    mean_galaxy_size=20, std_galaxy_size=5)

    model.draw()
