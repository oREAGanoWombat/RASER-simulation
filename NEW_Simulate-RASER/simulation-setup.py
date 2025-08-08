# Requirements
import os
import numpy as np
import random

# makedirs(path) function makes a new directory at location defined with `path` whenever it does not already exist
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

# _draw_circle() and _draw_ellipse() are helper functions that are used in generating random phantoms
def _draw_circle(image_array, center_x, center_y, radius, value):
    ny, nx = image_array.shape
    y, x = np.ogrid[0:ny, 0:nx]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # Ensure value does not exceed
    image_array[dist_from_center <= radius] = np.maximum(image_array[dist_from_center <= radius], value)

def _draw_ellipse(image_array, center_x, center_y, semi_axis_x, semi_axis_y, angle_rad, value):
    ny, nx = image_array.shape
    y, x = np.ogrid[0:ny, 0:nx]

    # Translate to origin, rotate, then apply ellipse equation
    x_translated = x - center_x
    y_translated = y - center_y

    x_rotated = x_translated * np.cos(angle_rad) + y_translated * np.sin(angle_rad)
    y_rotated = -x_translated * np.sin(angle_rad) + y_translated * np.cos(angle_rad)

    # Ellipse equation: (x_rotated/a)^2 + (y_rotated/b)^2 <= 1
    # Added a small epsilon to denominators to prevent division by zero if semi-axis becomes zero
    ellipse_mask = ((x_rotated / (semi_axis_x + 1e-9))**2 + (y_rotated / (semi_axis_y + 1e-9))**2) <= 1
    # Ensure value does not exceed 1.0
    image_array[ellipse_mask] = np.maximum(image_array[ellipse_mask], value)

# create_square_inversion_map(shape=(10,10)) creates a uniform inversion map of a solid square and a black border
def create_square_inversion_map(shape=(10,10)):
    square_size = (shape[0] // 2, shape[1] // 2)
    inversion_map = np.full(square_size, 1)
    inversion_map = np.pad(inversion_map, ((shape[0] // 4, shape[0] // 4), (shape[1] // 4, shape[1] // 4)), mode='constant', constant_values=0)
    return inversion_map