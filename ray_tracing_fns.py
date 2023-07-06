"""This module contains functions for ray tracing through a magnetic field space."""
from datetime import datetime
import numpy as np
import cupy as cp
import config
import config_fns

def make_ray(b_field, start_pos, dir_vec):
    """Simulates a ray of neutron passing through the field space.
    Returns the magnetic field values of each voxel the ray passes through.
    Precondition: 
        1. the ray does not pass through the edge of the space;
        2. the ray originates from one side of the space, and ends up on the other side.

    Args:
        b_field (np.array): The magnetic field in the space.
        start_pos (tuple): The starting (y, z) position of the ray in 2d space.
        dir_vec (tuple): The direction of the ray in terms of theta and phi.

    Returns:
        np.array: The magnetic field values of each voxel the ray passes through.
    """
    theta, phi = dir_vec

    theta = np.radians(theta)
    phi = np.radians(phi)

    deviate_angle = np.arctan(np.tan(theta) * np.cos(phi))
    if deviate_angle > config.MAX_ANGLE:
        raise ValueError("Max angle exceeded.")

    y_ray = np.sin(theta) * np.sin(phi)
    z_ray = np.sin(theta) * np.cos(phi)

    path_field = []
    field_path = []
    for x_pos, _ in enumerate(b_field):
        y_pos = int(round(float(y_ray * x_pos + start_pos[0])))
        z_pos = int(round(float(z_ray * x_pos + start_pos[1])))

        y_out_of_bound = (y_ray * len(b_field) + start_pos[0]) >= len(b_field[0])
        z_out_of_bound = (z_ray * len(b_field) + start_pos[1]) >= len(b_field[0][0])

        if y_pos < 0 or y_out_of_bound or z_pos < 0 or z_out_of_bound:
            return None
        else:
            field_at_loc = b_field[x_pos][y_pos][z_pos]
            ray_loc = [x_pos, y_pos, z_pos]

            path_field.append(field_at_loc)
            field_path.append(ray_loc)

        if y_pos < 0 or y_pos >= len(b_field[0]) or z_pos < 0 or z_pos >= len(b_field[0][0]):
            raise IndexError("Ray has left the space without reaching the end,"
                             "please check the starting position and direction. "
                             f"Ray last seen at (x, y, z) = {(x_pos, y_pos, z_pos)}. "
                             f"The final x position should be {len(b_field)}.") 

    try:
        return np.flip(path_field, axis=0), np.flip(field_path, axis=0)
    except TypeError:
        path_field = cp.array(path_field)
        field_path = cp.array(field_path)
        return cp.flip(path_field, axis=0), cp.flip(field_path, axis=0)

def ray_tracing_sim(field, pixel_pos, source):
    """Casts a spotlight from a camera pixel to the source to find the pixels within the spotlight.
    For each source pixel in the spotlight, find the direction of the ray and run make_ray() to get
        the field values along the ray as well as the ray's path. Calcualte the polarization rate of
        the ray, and the resulting intensity, and sum the intensities together to get the intensity 
        of the pixel.
    """
    # Create a background for the spotlight to cast on
    space_dim = np.shape(field)[:3]

    source_shape = np.shape(source)
    source_indx = np.indices(source_shape)

    # Find the critical angle for the spotlight
    deviant_radius = np.tan(np.radians(config.MAX_ANGLE)) * space_dim[0]

    # Isolate a circular area within deviant_radius from pixel_pos
    spotlight = np.where(np.sqrt((source_indx[0] - pixel_pos[0])**2 + (source_indx[1] - pixel_pos[1])**2) <= deviant_radius)

    # Iterate through each pixel in the spotlight
    i_pix = 0

    task_num = np.shape(spotlight)[1]
    ray_time = 0
    for i in range(task_num):
        ray_start = datetime.now()
        # print(f"Making ray {i} out of {task_num}. Time elapsed: {ray_time}.")
        source_y, source_z = spotlight[0][i], spotlight[1][i]
        source_val = source[source_y][source_z]

        # Get the direction of the ray
        theta = np.arctan((source_y - pixel_pos[0]) / space_dim[0])
        phi = np.arctan((source_z - pixel_pos[1]) / space_dim[0])

        dir_vec = (theta, phi)

        # Get the field values along the ray
        ray = make_ray(field, pixel_pos, dir_vec)
        if ray is None:
            continue
        else:
            path_field, field_path = ray

        # Get each ray's polarization
        p_ray = config_fns.find_polarization(path_field, config.WAVELENGTH)
        # NOTE: for now, the polarizatin is a scalar. In the future, it could be a vector.

        # has_nan = len(np.where((p_ray == np.nan))[0]) == 0
        # if has_nan is True:
            # continue

        # Get the intensity of the ray
        p_ray = np.linalg.norm(p_ray)

        i_ray = source_val * 0.5 * (p_ray - 1)

        i_pix += i_ray

        ray_time = datetime.now() - ray_start

    # rng = np.random.rand()
    # i_pix += rng

    return i_pix
