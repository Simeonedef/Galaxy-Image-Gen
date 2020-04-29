import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_dir = os.path.join('..', 'data', 'scored')


class Cluster:
    """
        Represents a continuous block of pixels on non-black color.
        Precise definition of 'non-black' is given with a greyscale threshold
    """

    def __init__(self, img, pixels):
        self.img = img
        self.pixels = pixels

        pixel_x, pixel_y = zip(*pixels)
        index_of_center_pixel = np.argmax(img[pixel_x, pixel_y])
        self.center_pixel = pixels[index_of_center_pixel]
        self.intensity = self.img[self.center_pixel]

    def size(self):
        """
        @return: number of pixels in cluster
        @rtype: int
        """
        return len(self.pixels)

    def get_center_pixel(self):
        '''
        @return: the center (brightest) pixel of the cluster
        @rtype: pair
        '''
        return self.center_pixel

    def get_intensity(self):
        '''
        @return: the greyscale color of the brightest pixel
        @rtype: float
        '''
        return self.intensity

    @staticmethod
    def find_clusters(img, pixels):
        """
        @param img: a grayscale image from the dataset
        @type img: 2D np array
        @param coords: the pixels from img which are not background
        @type coords: list of pairs
        @return: list of clusters formed by the input pixels
        @rtype: list of Cluster objects
        """
        assert img.ndim == 2

        q = []
        vis = set()
        pixels_set = set(pixels)
        dirx = [1, 1,  1, -1, -1 , -1, 0,  0]
        diry = [1, -1, 0,  1, -1,   0, 1, -1]

        clusters = []
        for pixel in pixels:
            if pixel in vis:
                continue

            curr_cluster = []
            q.append(pixel)
            while len(q) > 0:
                x, y = q.pop(0)
                if (x, y) in vis:
                    continue
                vis.add((x, y))
                curr_cluster.append((x, y))

                for deltax, deltay in zip(dirx, diry):
                    newx, newy = x + deltax, y + deltay

                    if (newx, newy) in vis or \
                            newx < 0 or newy < 0 or \
                            newx >= img.shape[0] or \
                            newy >= img.shape[1] or \
                            (newx, newy) not in pixels_set:
                        continue
                    q.append((newx, newy))

            clusters.append(curr_cluster)

        return [Cluster(img, cluster_pixels) for cluster_pixels in clusters]


def get_galaxies(img, threshold=40):
    # find all pixels which are not pitch black
    print(np.where(img != 0))
    print(img.shape)
    galaxy_x, galaxy_y = tuple(np.where(img > threshold))

    # list of pairs representing the galaxy pixel coords
    galaxy_coords = list(zip(galaxy_x, galaxy_y))

    galaxy_pixel_values = img[galaxy_x, galaxy_y]

    return galaxy_coords, galaxy_pixel_values


def get_background(img):
    background_x, background_y = np.where(img == 0)

    return background_x, background_y


def generate_statistics(data_dir, n_images=10):
    for ind, filename in enumerate(os.listdir(data_dir)):
        if ind == n_images:
            break
        img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)

        galaxy_coords, galaxy_pixel_values = get_galaxies(img)
        background_x, background_y = get_background(img)
        num_background_pixels = len(background_x)

        if len(galaxy_coords) > 10000:
            print(galaxy_pixel_values)
            print(img)

        print(filename, len(galaxy_coords), num_background_pixels)


if __name__ == "__main__":
    # just testing the Cluster class
    img = cv2.imread(os.path.join(data_dir, '1010772.png'), cv2.IMREAD_GRAYSCALE)
    galaxy_coords, _ = get_galaxies(img)
    clusters = Cluster.find_clusters(img, galaxy_coords)
    for cluster in clusters:
        print(cluster)
        print(cluster.size())
        print(cluster.get_center_pixel())
        print(cluster.get_intensity())
