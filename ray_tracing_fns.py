"""This module contains functions for ray tracing through a magnetic field space."""
from datetime import datetime
import numpy as np
import cupy as cp
import config
import config_fns
import main

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
        path_field: The magnetic field values of each voxel the ray passes through,
            in the chronological order of the ray's path.
        field_path: The (y, z) position of the ray in each x increment, also in the
            chronological order of the ray's path.
    """
    theta, phi = dir_vec # Radians

    y_increment = np.sin(theta)
    z_increment = np.sin(phi)
    x_tot = len(b_field)

    # Determine if the ray deviates too much
    deviant_angle = np.arctan((y_increment**2 + z_increment**2)**0.5)

    if deviant_angle > np.radians(config.MAX_ANGLE):
        err_msg = f"Deviant angle {np.degrees(deviant_angle)} "
        err_msg += f"is greater than the maximum deviant angle {config.MAX_ANGLE}."
        raise ValueError(err_msg)

    # Determine the final position of the ray
    y_ray_end = y_increment * x_tot + start_pos[0]
    z_ray_end = z_increment * x_tot + start_pos[1]

    y_out_of_bound = y_ray_end >= len(b_field[0])-1 or y_ray_end < 0
    z_out_of_bound = z_ray_end >= len(b_field[0][0])-1 or z_ray_end < 0

    # If out of bound, print message and return None (move on to next ray)
    if y_out_of_bound or z_out_of_bound:
        return None

    path_field = []
    field_path = []
    for x_pos, _ in enumerate(b_field):
        # For each x position, determine the y-z position of the ray #NOTE: this step is fine
        y_ray_pos = np.float64(y_increment * x_pos + start_pos[0])
        z_ray_pos = np.float64(z_increment * x_pos + start_pos[1])

        # Determine if the ray is out of bound
        y_ray_finish = len(b_field) * y_increment + start_pos[0]
        z_ray_finish = len(b_field) * z_increment + start_pos[1]

        y_out_of_bound = y_ray_finish >= len(b_field[0])-1 or y_ray_finish < 0
        z_out_of_bound = z_ray_finish >= len(b_field[0][0])-2 or z_ray_finish < 0

        if y_out_of_bound or z_out_of_bound:
            return None
        else:
            y_pos = int(round(y_ray_pos))
            z_pos = int(round(z_ray_pos))
            # print(f"z_pos = {z_pos}, z_ray_finish = {z_ray_finish}")
            field_at_loc = b_field[x_pos][y_pos][z_pos]
            ray_loc = [x_pos, y_ray_pos, z_ray_pos]

            path_field.append(field_at_loc)
            field_path.append(ray_loc)

    try:
        return np.flip(path_field, axis=0), np.flip(field_path, axis=0)
    except TypeError:
        path_field = cp.array(path_field)
        field_path = cp.array(field_path)
        return cp.flip(path_field, axis=0), cp.flip(field_path, axis=0)

def ray_tracing_sim(field, pixel_pos, source, wavelength, initial_polarization):
    """Casts a spotlight from a camera pixel to the source to find the pixels within the spotlight.
    For each source pixel in the spotlight, find the direction of the ray and run make_ray() to get
        the field values along the ray as well as the ray's path. Calcualte the polarization rate of
        the ray, and the resulting intensity, and sum the intensities together to get the intensity 
        of the pixel.

    Args:
        field (np.array): The magnetic field in the space.
        pixel_pos (tuple): The position of the pixel in 2d space.
        source (np.array): The source of the neutrons.

    Returns:
        float: The intensity of the pixel.
    """
    # Create a background for the spotlight to cast on
    space_dim = np.shape(field)[:3]

    source_shape = np.shape(source)
    source_indx = np.indices(source_shape)

    # Find the critical angle for the spotlight
    deviant_radius = np.tan(np.radians(config.MAX_ANGLE)) * space_dim[0]

    # Isolate a circular area within deviant_radius from pixel_pos
    spotlight = np.where(((source_indx[0] - pixel_pos[0])**2 + (source_indx[1] - pixel_pos[1])**2)**0.5 <= deviant_radius)

    # Iterate through each pixel in the spotlight
    i_pix = np.float64(0)

    task_num = np.shape(spotlight)[1]
    ray_time = 0
    for i in range(task_num):
        # ray_start = datetime.now()
        # ray_time += (datetime.now() - ray_start).total_seconds()
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
        p_ray = config_fns.find_polarization(path_field, wavelength, initial_polarization)
        # NOTE: for now, the polarizatin is a scalar. In the future, it could be a vector.
        # p_ray = np.float64(p_ray%(np.pi * 2))

        i_ray = source_val * 0.5 * (1 - p_ray)

        i_pix += i_ray

    return i_pix
