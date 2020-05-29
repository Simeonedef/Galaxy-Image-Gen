import numpy as np


class BaselinePositionModel:
    """
        Assumes the image is divided in a grid. Generates the positions of the galaxies in a grid
        1) Samples the number of galaxies from a normal distribution.
        2) Samples the positions for the galaxies in the grid from a uniform distribution.
        This model will be used in a combination with a GAN model that generates a galaxy at the locations
        specified by this model.
    """
    def __init__(self, grid_size, mean_num_galaxies=20, std_num_galaxies=2):
        self.grid_size = grid_size
        self.mean_num_galaxies = mean_num_galaxies
        self.std_num_galaxies = std_num_galaxies

        self.num_galaxies = None  # to be sampled later

    def generate(self, n_images):
        return [self.draw() for _ in range(n_images)]

    def draw(self):
        self.sample_num_galaxies()
        galaxy_cells = self.sample_galaxy_cells()

        img = np.zeros(self.grid_size)
        for (x, y) in galaxy_cells:
            img[x][y] = 1

        return img

    def sample_num_galaxies(self):
        self.num_galaxies = max(0, int(np.random.normal(self.mean_num_galaxies, self.std_num_galaxies)))

    def sample_galaxy_cells(self):
        # self.sample_num_galaxies needs to be always called first
        assert self.num_galaxies is not None

        galaxy_cell_x = np.random.randint(0, self.grid_size[0], size=self.num_galaxies)
        galaxy_cell_y = np.random.randint(0, self.grid_size[1], size=self.num_galaxies)
        return zip(galaxy_cell_x, galaxy_cell_y)
