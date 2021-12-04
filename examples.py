"""
@author: The Absolute Tinkerer
"""

import os
import math
import time
import random

import numpy as np

from PIL import Image

from PyQt5.QtGui import QColor, QPen, QPixmap
from PyQt5.QtCore import QPointF, QRect

# Absolute Tinkerer imports
import painter
from utils import QColor_HSV, save, Perlin2D


def draw_white_noise(width, height, fname):
    assert not os.path.exists(fname), 'File already exists!'

    # Create a matrix of random values between zero and one
    pixels = np.random.random(size=(height, width))

    # Now modify the random values to be 0-255 (pixel color range)
    pixels = 255*pixels

    # The function to write the array of pixels to an image requires integers, not float values
    pixels = pixels.astype(np.uint8)

    # We choose to make random values grayscale, so each RGB element is identical. This code adds the third dimension
    # to our pixels array
    pixels = pixels[:, :, np.newaxis]

    # We need to repeat each value to finalize the pixels arrays in the grayscale space
    pixels = np.repeat(pixels, 3, axis=2)

    # Now create the image from an array of pixels
    im = Image.fromarray(pixels)

    # Save the image to file
    im.save(fname)


def draw_perlin(nx, ny, width, height, fname):
    assert not os.path.exists(fname), 'File already exists'

    # Initialize Perlin Noise
    noise = (Perlin2D(width, height, nx, ny) + 1)/2

    # Convert to pixels
    pixels = 255 * noise
    pixels = pixels.astype(np.uint8)
    pixels = pixels[:, :, np.newaxis]
    pixels = np.repeat(pixels, 3, axis=2)

    # Create and save the image from pixels
    im = Image.fromarray(pixels)
    im.save(fname)

    return noise


def draw_vectors(nx, ny, width, height, seed=random.randint(0, 100000000), flow_length=100, n_vectors=50):
    p_path = f'{seed}_1_perlin_noise.jpg'
    v_path = f'{seed}_2_vectors'
    f_path = f'{seed}_3_flow_field'

    # Ensure we don't overwrite paths
    assert not os.path.exists(p_path), 'Perlin Noise image already exists!'
    assert not os.path.exists(v_path), 'Vectors image already exists!'
    assert not os.path.exists(f_path), 'Flow field image already exists!'

    # Set the random seed for repeatability
    np.random.seed(seed)

    # Create the Perlin Noise image
    noise = draw_perlin(nx, ny, width, height, p_path)

    # Initialize the painter object for drawing
    p = painter.Painter(width, height)
    p.setRenderHint(p.Antialiasing)  # allow smooth drawing

    def draw_arrow(p, x_i, y_i, length=100, angle=0):
        # Compute the second points and draw the arrow body
        x_f = x_i + length*math.cos(math.radians(angle))
        y_f = y_i - length*math.sin(math.radians(angle))
        p.drawLine(x_i, y_i, x_f, y_f)

        # Compute the arrow head second points
        a_angle1, a_angle2 = math.radians(angle-30), math.radians(angle+30)
        x1 = x_f - (length/10)*math.cos(a_angle1)
        y1 = y_f + (length/10)*math.sin(a_angle1)
        x2 = x_f - (length/10)*math.cos(a_angle2)
        y2 = y_f + (length/10)*math.sin(a_angle2)
        p.drawLine(x_f, y_f, x1, y1)
        p.drawLine(x_f, y_f, x2, y2)

    # Load the Perlin Noise image and draw it with the painter
    p.drawPixmap(QRect(0, 0, width, height), QPixmap(p_path))

    # Now we're drawing red arrows for vectors, so set the pen color to red
    p.setPen(QColor(255, 0, 0))

    # We need arrow locations, so create a grid of n_vectors x n_vectors, excluding the image border
    _nx, _ny = n_vectors, n_vectors
    dx, dy = width / (_nx + 1), height / (_ny + 1)
    x_points = [dx + i*dx for i in range(_nx)]
    y_points = [dy + i*dy for i in range(_ny)]

    # Draw the arrows
    for x in x_points:
        for y in y_points:
            angle = 360*noise[int(x), int(y)]
            draw_arrow(p, x, y, length=min(dx, dy), angle=angle)

    # Save the vector image
    save(p, fname=v_path, folder='.')

    # End this painter, but don't exit
    p.endProgram(exit_code=1):

    # Now draw the flow field. Start by initializing a new painter
    p = painter.Painter(width, height)
    p.setRenderHint(p.Antialiasing)  # allow smooth drawing
    p.setPen(QColor(0, 0, 0))  # pen color set to black

    # Step size between points
    STEP_SIZE = 0.001 * max(width, height)

    # Draw the flow field
    for x in x_points:
        for y in y_points:
            # The starting position
            x_s, y_s = x, y
            # The current line length tracking variable
            c_len = 0
            while c_len < flow_length:
                # angle between 0 and 2*pi
                angle = 2 * noise[int(x_s), int(y_s)] * math.pi

                # Compute the new point
                x_f = x_s + STEP_SIZE * math.cos(angle)
                y_f = y_s - STEP_SIZE * math.sin(angle)

                # Draw the line
                p.drawLine(QPointF(x_s, y_s), QPointF(x_f, y_f))

                # Update the line length
                c_len += math.sqrt((x_f - x_s) ** 2 + (y_f - y_s) ** 2)

                # Break from the loop if the new point is outside our image bounds
                # or if we've exceeded the line length; otherwise update the point
                if x_f < 0 or x_f >= width or y_f < 0 or y_f >= height or c_len > flow_length:
                    break
                else:
                    x_s, y_s = x_f, y_f
    save(p, fname=f_path, folder='.')


def draw_flow_field(width, height):
    colors = [200, 140, 70, 340, 280]
    for i, mod in enumerate(colors):
        print('Starting Image %s/%s' % (i + 1, len(colors)))
        p = painter.Painter(width, height)

        # Allow smooth drawing
        p.setRenderHint(p.Antialiasing)

        # Draw the background color
        p.fillRect(0, 0, width, height, QColor(0, 0, 0))

        # Set the pen color
        p.setPen(QPen(QColor(150, 150, 225, 5), 2))

        num = 1
        for j in range(num):
            print('Creating Noise... (%s/%s)' % (j + 1, num))
            p_noise = Perlin2D(width, height, 2, 2)
            print('Noise Generated! (%s/%s)' % (j + 1, num))

            MAX_LENGTH = 2 * width
            STEP_SIZE = 0.001 * max(width, height)
            NUM = int(width * height / 1000)
            POINTS = [(random.randint(0, width - 1), random.randint(0, height - 1)) for i in range(NUM)]

            for k, (x_s, y_s) in enumerate(POINTS):
                print(f'{100 * (k + 1) / len(POINTS):.1f}'.rjust(5) + '% Complete', end='\r')

                # The current line length tracking variable
                c_len = 0

                # Actually draw the flow field
                while c_len < MAX_LENGTH:
                    # Set the pen color for this segment
                    sat = 200 * (MAX_LENGTH - c_len) / MAX_LENGTH
                    hue = (mod + 130 * (height - y_s) / height) % 360
                    p.setPen(QPen(QColor_HSV(hue, sat, 255, 20), 2))

                    # angle between -pi and pi
                    angle = 2 * p_noise[int(x_s), int(y_s)] * math.pi

                    # Compute the new point
                    x_f = x_s + STEP_SIZE * math.cos(angle)
                    y_f = y_s + STEP_SIZE * math.sin(angle)

                    # Draw the line
                    p.drawLine(QPointF(x_s, y_s), QPointF(x_f, y_f))

                    # Update the line length
                    c_len += math.sqrt((x_f - x_s) ** 2 + (y_f - y_s) ** 2)

                    # Break from the loop if the new point is outside our image bounds
                    # or if we've exceeded the line length; otherwise update the point
                    if x_f < 0 or x_f >= width or y_f < 0 or y_f >= height or c_len > MAX_LENGTH:
                        break
                    else:
                        x_s, y_s = x_f, y_f

            save(p, fname=f'image_{mod}_{num}', folder='.', overwrite=True)
