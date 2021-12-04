"""
@author: The Absolute Tinkerer
"""

import os
import math
import numpy as np

from PyQt5.QtGui import QColor


def QColor_HSV(h, s, v, a=255):
    """
    Hue        : > -1 [wraps between 0-360]
    Saturation : 0-255
    Value      : 0-255
    Alpha      : 0-255
    """
    color = QColor()
    color.setHsv(*[int(e) for e in [h, s, v, a]])
    return color


def save(p, fname='image', folder='Images', extension='jpg', quality=100, overwrite=True):
    if not os.path.exists(folder):
        os.mkdir(folder)

    # The image name
    imageFile = f'{folder}/{fname}.{extension}'

    # Do not overwrite the image if it exists already
    if os.path.exists(imageFile):
        assert overwrite, 'File exists and overwrite is set to False!'

    # fileName, format, quality [0 through 100]
    p.saveImage(imageFile, imageFile[-3:], quality)


def Perlin2D(width, height, n_x, n_y, clampHorizontal=False, clampVertical=False):
    """
    Constructor

    Optimizations were gained from studying:
    https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py

    Parameters:
    -----------
    width : int
        The width of the canvas
    height : int
        The height of the canvas
    n_x : int
        The number of x tiles; must correspond to an integer x-edge length
    n_y : int
        The number of y tiles; must correspond to an integer y-edge length
    clampHorizontal : boolean
        Imagine the Perlin Noise on a sheet of paper - form a cylinder with
        the horizontal edges. If True, cylinder will be continuous noise
    clampVertical : boolean
        Imagine the Perlin Noise on a sheet of paper - form a cylinder with
        the vertical edges. If True, cylinder will be continuous noise

    Returns:
    --------
    <value> : numpy array
        noise values for array[width, height] between -1 and 1
    """
    # First ensure even number of n_x and n_y divide into the width and height,
    # respectively
    msg = 'n_x and n_y must evenly divide into width and height, respectively'
    assert width % n_x == 0 and height % n_y == 0, msg

    # We start off by defining our interpolation function
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    # Next, we generate the gradients that we are using for each corner point
    # of the grid
    angles = 2 * np.pi * np.random.rand(n_x + 1, n_y + 1)
    r = math.sqrt(2)  # The radius of the unit circle
    gradients = np.dstack((r * np.cos(angles), r * np.sin(angles)))

    # Now, if the user has chosen to clamp at all, set the first and last row/
    # column equal to one another
    if clampHorizontal:
        gradients[-1, :] = gradients[0, :]
    if clampVertical:
        gradients[:, -1] = gradients[:, 0]

    # Now that gradient vectors are complete, we need to create the normalized
    # distance from each point to its starting grid point. In other words, this
    # is the normalized distance from the grid tile's origin based upon the
    # grid tile's width and height
    delta = (n_x / width, n_y / height)
    grid = np.mgrid[0:n_x:delta[0], 0:n_y:delta[1]].transpose(1, 2, 0) % 1

    # At this point, we need to compute the dot products for each corner of the
    # grid. To do this, we first need proper-dimensioned gradient vectors - do
    # this now. A computation for number of points per tile is needed as well
    px, py = int(width / n_x), int(height / n_y)
    gradients = gradients.repeat(px, 0).repeat(py, 1)
    g00 = gradients[:-px, :-py]
    g10 = gradients[px:, :-py]
    g01 = gradients[:-px, py:]
    g11 = gradients[px:, py:]

    # Compute dot products for each corner
    d00 = np.sum(g00 * grid, 2)
    d10 = np.sum(g10 * np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])), 2)
    d01 = np.sum(g01 * np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)), 2)
    d11 = np.sum(g11 * np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)), 2)

    # We're doing improved perlin noise, so we use a fade function to compute
    # the x and y fractions used in the linear interpolation computation
    # t is the faded grid
    # u is the faded dot product between the top corners
    # v is the faded dot product between the bottom corners
    # _x and _y are the fractional (0-1) location of x, y in the tile
    t = fade(grid)
    u = d00 + t[:, :, 0] * (d10 - d00)
    v = d01 + t[:, :, 0] * (d11 - d01)

    # Now perform the second dimension's linear interpolation to return value
    return u + t[:, :, 1] * (v - u)
