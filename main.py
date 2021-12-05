"""
@author: The Absolute Tinkerer
"""

import os
import random

from examples import draw_flow_field, draw_delta_body, draw_white_noise, draw_perlin, draw_vectors, draw_perlin_rounding


if __name__ == '__main__':
    output_folder = 'Images'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # draw_flow_field(6000, 4000)
    # draw_white_noise(600, 300, f'{output_folder}/white_noise.jpg')
    # draw_perlin(5, 5, 1000, 1000, 'output_image.jpg')
    # draw_vectors(5, 5, 1000, 1000)
    # draw_perlin_rounding(6000, 4000, 'perlin_rounding')
    # draw_delta_body(2000, 2000, mode='noise')
