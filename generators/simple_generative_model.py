import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_radius(size):
    """
        Roughly estimates the 'radius' of a galaxy given its size i.e the number of pixels.
        Assumes the shape of a galaxy is a rotated square, which is an approximation of the actual shape which is
        an imperfect curvy rhombus.
        In this approximation, the number of pixels is the area of this square, and the
        radius is half the diagonal.
    """
    side = np.sqrt(size)  # derive side of square from the area of the square
    d = np.sqrt(2) * side  # derive diagonal from square side
    r = int(d / 2)

    return r


def draw_galaxy(img, center, radius, intensity, fade='auto'):
    """
    Draws a galaxy of the following fixed shape
          *
         ***
        *****
         ***
          *
    @param img: image to draw the galaxy in
    @type img: 2D np.array
    @param center: center pixel of the galaxy
    @type center: pair
    @param size: number of pixels in galaxy
    @type size: int
    @param intensity: peak intensity (at the center) of the galaxy
    @type intensity: int (0-255)
    @param fade: the amount by which the intensity is reduced per pixel as we move away from the center pixel
    @type fade: int (0-255) or 'auto'
    """

    # draw upper half, including center row
    for x_delta in range(radius):
        x = center[0] - x_delta
        if x < 0:
            break

        # center and to the left
        for y_delta in range(radius - x_delta):
            y = center[1] - y_delta
            if y < 0:
                break
            img[x][y] = max(int(intensity - fade * y_delta - fade * x_delta), 0)

        # to the right
        for y_delta in range(1, radius - x_delta):
            y = center[1] + y_delta
            if y < 0 or y >= img.shape[1]:
                break

            img[x][y] = max(int(intensity - fade * y_delta - fade * x_delta), 0)

    # draw lower half
    for x_delta in range(1, radius + 1):  # for each row
        x = center[0] + x_delta
        if x >= img.shape[0]:
            break

        # center and to the left
        for y_delta in range(radius - x_delta):  # for each column
            y = center[1] - y_delta
            if y < 0 or y >= img.shape[1]:
                break
            img[x][y] = max(intensity - fade * y_delta - fade * x_delta, 0)

        # to the right
        for y_delta in range(radius - x_delta):
            y = center[1] + y_delta
            if y < 0 or y >= img.shape[1]:
                break

            img[x][y] = max(intensity - fade * y_delta - fade * x_delta, 0)


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
    def __init__(self, mean_num_galaxies=8, std_num_galaxies=2,
                       mean_galaxy_size=15, std_galaxy_size=10,
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
