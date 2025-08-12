"""
utility_functions.py
Contains general utility functions for directory management and creating sample inversion maps (phantoms) for RASER simulation
"""

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

# create_sample_inversion_map(shape=(10,10)) creates a uniform inversion map of a specified phantom type
def create_sample_inversion_map(shape=(10,10), inversion_map_mode='square'):
    ny, nx = shape
    inversion_map = np.zeros(shape, dtype=np.float64) # creates blank canvas

    if inversion_map_mode == 'square':
        # default square size is 1/2 minimum dimension
        square_size = min(ny, nx) // 2
        start_x = (nx - square_size) // 2
        end_x = start_x + square_size
        start_y = (ny - square_size) // 2
        end_y = start_y + square_size
        inversion_map[start_y:end_y, start_x:end_x] = 1.0
    elif inversion_map_mode == 'circle':
        center_x, center_y = nx // 2, ny // 2
        radius = min(nx, ny) // 3
        _draw_circle(inversion_map, center_x, center_y, radius, 1.0)
    elif inversion_map_mode == 'ellipse':
        center_x, center_y = nx // 2, ny // 2
        semi_axis_x = nx // 3.5  # Adjusted for better fit
        semi_axis_y = ny // 5
        angle_rad = np.pi / 4  # 45 degrees rotation
        _draw_ellipse(inversion_map, center_x, center_y, semi_axis_x, semi_axis_y, angle_rad, 1.0)
    elif inversion_map_mode == 'random':
        num_blobs = random.randint(2, 6)  # Generate between 2 and 6 random blobs
        for _ in range(num_blobs):
            blob_type = random.choice(['circle', 'ellipse'])
            # Random center within reasonable bounds to keep blobs mostly visible
            center_x = random.randint(nx // 5, 4 * nx // 5)
            center_y = random.randint(ny // 5, 4 * ny // 5)
            # Random intensity for each blob, creating varying population inversion profiles
            intensity = random.uniform(0.2, 1.0)

            if blob_type == 'circle':
                # Random radius within reasonable bounds
                radius = random.randint(min(nx, ny) // 12, min(nx, ny) // 6)
                _draw_circle(inversion_map, center_x, center_y, radius, intensity)
            elif blob_type == 'ellipse':
                # Random semi-axes and rotation for ellipses
                semi_axis_x = random.randint(min(nx, ny) // 12, min(nx, ny) // 5)
                semi_axis_y = random.randint(min(nx, ny) // 12, min(nx, ny) // 5)
                angle_rad = random.uniform(0, np.pi)
                _draw_ellipse(inversion_map, center_x, center_y, semi_axis_x, semi_axis_y, angle_rad, intensity)
    else:
        raise ValueError("Invalid phantom_type. Must be 'square', 'circle', 'ellipse', or 'random_blobs'.")

    # Normalize the entire map to have its maximum value at 1.0, preserving relative intensities
    if np.max(inversion_map) > 0:
        inversion_map = inversion_map / np.max(inversion_map)

    return inversion_map