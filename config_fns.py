"""Utility functions that are not directly related to the simulation.
    Editing this file is not advised.
"""
import numpy as np
import config

def gaussian_2d(space_dim, sigma, mu_):
    """Returns a 2D Gaussian distribution in a 2D array.
    
    Args:
        sigma (tuple): The standard deviation of the Gaussian distribution in both y
            and z directions.
        mu (tuple): The mean of the Gaussian distribution in both y and z directions.

    Returns:
        np.array: The 2D Gaussian distribution.
    """
    sigma_y, sigma_z = sigma
    mu_y, mu_z = mu_

    mu_y = int(mu_y * space_dim[1])
    mu_z = int(mu_z * space_dim[2])

    sigma_y = int(sigma_y * space_dim[1])
    sigma_z = int(sigma_z * space_dim[2])

    output = np.zeros((space_dim[1], space_dim[2]))

    for y_pos, _ in enumerate(output):
        for z_pos, _ in enumerate(output[y_pos]):
            output_temp_y = np.exp(-np.power(y_pos-mu_y, 2.)/(2*np.power(sigma_y, 2.)))
            output_temp_z = np.exp(-np.power(z_pos-mu_z, 2.)/(2*np.power(sigma_z, 2.)))
            output_temp = output_temp_y * output_temp_z

            output[y_pos][z_pos] = output_temp

    return output

def dist(p_1, p_2):
    """Calculates the distance between two points."""
    return ((p_1[0]-p_2[0])**2 + (p_1[1]-p_2[1])**2)**0.5

def read_data(data_path, filename):
    """Reads data from COMSOL output file.
    
    Args:
        data_path (str): Path to the data file.
        filename (str): Name of the data file.
        facing_dir (str): Direction in which the neutron is travelling. Defaults to "x".

    Returns:
        field (np.ndarray): Magnetic field data.
        field_loc (np.ndarray): Location of the magnetic field data.
    """
    with open(data_path + filename, 'r', encoding="utf-8") as f:
        data = []
        for _, line in enumerate(f):
            row = line.split("  ")

            if row[0][0] == "%":
                row[0] = row[0][2:]
            else:
                if isinstance(last_row[0], float) is False:
                    pass

            row[-1] = row[-1][:-1]

            try:
                row = [np.float64(i) for i in row if i != '']
                data.append(row)
            except ValueError:
                row = [i.strip() for i in row if i != '']

            last_row = row

    data = np.transpose(data)

    field_input = np.array([data[5], data[4], data[3]])

    field_loc = np.array([data[2], data[1], data[0]])

    for i, row in enumerate(field_loc):
        temp_row = row - min(row)

        next_min = list(set(temp_row))[1]
        temp_row = temp_row / next_min

        field_loc[i] = temp_row
    
    space_dim = (len(set(field_loc[0])), len(set(field_loc[1])), len(set(field_loc[2])))
    space = np.transpose(np.indices(space_dim), (1, 2, 3, 0))

    field = np.copy(space)
    field = field.astype(np.float64)

    field_loc = np.transpose(field_loc)
    field_input = np.transpose(field_input)

    for i, vec in enumerate(field_loc):
        x_pos, y_pos, z_pos = vec.astype(int)
        field_temp = np.copy(field_input[i])
        field[x_pos, y_pos, z_pos] = field_temp

    return field, field_loc

def find_polarization(path_field, wavelength):
    """Calculates a neutron's polarization rate after passing through a magnetic field along its path.
    
    Args:
        path_field (np.array): The magnetic field values of each voxel the ray passes through.
        wavelength (float): The wavelength of the neutron.

    Returns:
        float: The polarization rate of the neutron.
    """
    GAMMA = 1.83247179 * 10**8
    h = 6.626e-34
    m = 1.675e-27
    field_integral = np.sum(path_field, axis=0) # In Tesla
    temp = GAMMA * m / h * field_integral
    P = 1 - temp**2 * wavelength**2

    return P


