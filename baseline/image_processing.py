import os
from collections import Counter, defaultdict

import cv2
import numpy as np

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

    def num_intensities(self):
        """
        @return: number of different intensities in the cluster pixels
        @rtype: int
        """
        pixel_x, pixel_y = zip(*self.pixels)
        return np.unique(self.img[pixel_x, pixel_y].flatten(), return_counts=True)[0].shape[0]

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


def get_galaxy_pixels(img, threshold=40):
    """
    @param img: greyscale image of a real galaxy, noisy galaxy or a completely fake image
    @type img: 2D numpy array
    @param threshold: pixels above threshold are considered a part of a galaxy
    @type threshold: int, [0, 255]
    @return: the coordinates of the galaxy pixels and their intenstity
    @rtype: pair of lists
    """
    # find all pixels which are above threshold
    galaxy_x, galaxy_y = tuple(np.where(img > threshold))

    # list of pairs representing the galaxy pixel coords
    galaxy_coords = list(zip(galaxy_x, galaxy_y))

    galaxy_pixel_values = img[galaxy_x, galaxy_y]

    return galaxy_coords, galaxy_pixel_values


def get_background(img):
    background_x, background_y = np.where(img == 0)

    return background_x, background_y


# def estimate_background_intensity_threshold(img, background_pixels_ratio=0.8):
#     pixel_intesity_cnts = pixel_intensity_histogram(img)
#     num_total_pixels = img.shape[0] * img.shape[1]
#
#     num_background_pixels = 0
#     for intensity, cnt in pixel_value_cnts.values():
#         num_background_pixels += cnt
#         if num_background_pixels / num_total_pixels > background_pixels_ratio:
#             break


def estimate_background_intensity_threshold(img, background_pixels_ratio=0.8):
    """
    Start from the darkest pixel (intensity = 0) and iteratively increase the intensity, adding the pixels with
    that intensity to 'background'. We stop when the number of pixels will reach some threshold (like 80% of total
    pixels). Undefined behavior for fake/noise images.
    @param img: a greyscale image
    @type img: 2D numpy array
    @param background_pixels_ratio: the ratio of background vs all pixels when we stop the iteration
    @type background_pixels_ratio: float, [0, 1]
    @return: the pixel intensity such that pixels of <= intensity occupy at least background_pixels_ratio of the image
    @rtype: int, [0, 255]
    """
    assert img.ndim == 2

    pixel_intesity_cnts = pixel_intensity_histogram(img)
    num_total_pixels = img.shape[0] * img.shape[1]

    num_background_pixels = 0
    for intensity in range(0, 256):
        num_background_pixels += pixel_intesity_cnts[intensity]
        if num_background_pixels / num_total_pixels > background_pixels_ratio:
            return intensity


def pixel_intensity_histogram(img):
    """
    @param img: a greyscale image
    @type img: 2d np.array
    @return: a mapping of pixel intesity -> cnt of pixels with that intensity in img
    @rtype: dict
    """
    return defaultdict(int, Counter(img.flatten()))


if __name__ == "__main__":
    # just testing the Cluster class
    img = cv2.imread(os.path.join(data_dir, '1013618.png'), cv2.IMREAD_GRAYSCALE)
    pixel_value_cnts = Counter(img.flatten())

    print(pixel_value_cnts)
