import numpy as np


def draw_small_galaxy(img, center, size, intensity):
    """
    Draws small galaxies (num pixels < 5). This is a separate method since these small clusters cannot have the diamond shape
    of larger clusters.
    @param img: image to draw the galaxy in
    @type img: 2D np.array
    @param center: center pixel of the galaxy
    @type center: pair
    @param size: num of pixels in small galaxy
    @type size: int
    @param intensity: peak intensity (at the center) of the galaxy
    @type intensity: int (0-255)
    """
    if size == 1:
        img[center[0], center[1]] = intensity
    if size == 2:
        img[center[0], max(0, center[1]-1)] = 200
    if size == 3:
        img[min(img.shape[0]-1, center[0] + 1), center[1]] = 150
    if size == 4:
        img[min(img.shape[0]-1, center[0] + 1), max(0, center[1]-1)] = intensity


def draw_galaxy(img, center, radius, intensity, fade=5):
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
    @param radius: radius of galaxy (in pixels)
    @type radius: int
    @param intensity: peak intensity (at the center) of the galaxy
    @type intensity: int (0-255)
    @param fade: the amount by which the intensity is reduced per pixel as we move away from the center pixel
    @type fade: int (0-255)
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


def get_radius(size):
    """
        Roughly estimates the 'radius' of a galaxy given its size i.e the number of pixels.
        Assumes the shape of a galaxy is a rotated square, which is an approximation of the actual shape which is
        an imperfect 'curvy' rhombus.
        In this approximation, the number of pixels is the area of this square, and the
        radius is half the diagonal.
    """
    side = np.sqrt(size)  # derive side of square from the area of the square
    d = np.sqrt(2) * side  # derive diagonal from square side
    r = int(d / 2)

    return r