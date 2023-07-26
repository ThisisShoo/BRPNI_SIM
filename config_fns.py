"""Utility functions that are not directly related to the simulation.
    Editing this file is not advised.
"""
import main
import multiprocessing as mp
from datetime import datetime
import os
import numpy as np
from matplotlib import pyplot as plt
import config
import ray_tracing_fns

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

def find_polarization(path_field, wavelength, initial_polarization):
    """Calculates a neutron's polarization rate after passing through a magnetic field along 
        its path.

    Args:
        path_field (np.array): The magnetic field values of each voxel the ray passes through.

    Returns:
        float: The polarization rate of the neutron.
    """
    GAMMA = 1.83247171 * 10**8 # s^-1 T^-1
    h = 6.626 * 10**-34 # J*s
    m = 1.674 * 10**-27 # kg
    field_integral = np.sum(path_field, axis=0) # In the same unit as COMSOL's export

    dP_temp = (GAMMA * m / h * np.linalg.norm(field_integral) * wavelength)**2

    dP = max(dP_temp, 0)

    P = (1 - dP) * initial_polarization

    return P

def master_execution(data_path, data_file, source_profile, axis, initial_polarization,
                      wavelength, plot_name, plot_field_first, show_progress, make_plot):
    """Master execution function.
    
    Args:
        data_path (str): Path to the data file.
        data_file (str): Name of the data file.
        source_profile (str): Type of source profile to be used.
        axis (str): Axis to be raytraced along.
        initial_polarization (float): Initial polarization rate of the neutron in decimals.
        wavelength (float): Wavelength of the neutron in Angstroms.
        plot_name (str): Name of the plot.
        plot_field_first (bool): If true, generates a 3D plot of the field before raytracing.
        show_progress (bool): If true, prints the progress.
        make_plot (bool): If true, generates a plot of the raytracing result.

    Returns:
        None
    """
    # Identify input data type
    if data_file.rsplit('.', maxsplit=1)[-1] == 'txt':
        field, _ = read_data(data_path, data_file)
        field_loc = np.indices(np.shape(field)[:-1])
    elif data_file.rsplit('.', maxsplit=1)[-1] == 'npz':
        data = np.load(data_path + "/" + data_file)
        field = data["field"]
        field_loc = data["field_loc"]

    field = np.transpose(field, (3, 0, 1, 2))

    # Rotate the field (40, 32, 24, 3) (z, y, x, 3)
    axis_rot = {"x": (3, 2, 1, 0), "y": (2, 3, 1, 0), "z": (1, 3, 2, 0)}

    field_loc = np.transpose(field_loc, axis_rot[axis])
    field = np.transpose(field, axis_rot[axis])

    space_dim = np.array(np.shape(field)[0:3])
    space_dim = np.array(space_dim)

    field_loc_t = np.transpose(field_loc, (3, 0, 1, 2))
    field_t = np.transpose(field, (3, 0, 1, 2))

    # Plot the input data
    if plot_field_first:
        ax = plt.figure().add_subplot(projection='3d')
        x, y, z = field_loc_t
        ax.quiver(x, y, z,
                  field_t[2], field_t[1], field_t[0],
                  length = 1, normalize = True)
        plt.show()

    if source_profile is None:
        source = np.ones((space_dim[1], space_dim[2]))
        print("No source profile specified, using a uniform source profile.")
    elif source_profile == "gaussian":
        source = gaussian_2d(space_dim,
                                        config.SOURCE_STD,
                                        config.SOURCE_NORM)
    else:
        pass

    # Do raytracing
    start_time = datetime.now()
    print(f"Starting raytracing at {str(start_time)}, along axis {axis}.")

    output = np.zeros(space_dim[1:3])

    tot_row = np.shape(output)[0]
    with mp.Pool(processes = os.cpu_count()) as pool:
        for i, row in enumerate(output):
            row_start = datetime.now()
            progress = round(i/tot_row * 100, 2)
            for j, col in enumerate(row):
                i_pix = pool.apply_async(ray_tracing_fns.ray_tracing_sim,
                                         args=(field, (i, j), source,
                                               wavelength, initial_polarization))
                i_pix = i_pix.get()

                output[i][j] = i_pix

            row_time = datetime.now() - row_start
            if show_progress:
                total_time = datetime.now() - start_time
                print(f"Processed row {i}/{tot_row}, progress {progress}"
                      f"%. Last row used {row_time}. "
                      f"Total time elapsed: {total_time}. "
                      f"Estimated time remaining: {total_time/(i + 1) * (tot_row - i)}. ")

    completion = datetime.now()
    print(f"{completion} - Processing complete."
          f"Total time taken: {completion - start_time}")

    # Compute the sum of the output, and store the result in a file with settings.
    output_sum = np.sum(output)
    with open(f"Data_Folder/{plot_name}.txt", 'a', encoding='utf8') as f:
        printout = f"\u03BB:{wavelength}\u212B, P0:{initial_polarization*100}%, "
        printout += f"intensity:{output_sum}\n"
        f.write(printout)

        print("Result saved.")

    if make_plot is False:
        return None

    plot_axis = "zyx".replace(axis, '')

    # Make the plot
    size = (space_dim[2] + 1.5, space_dim[1])/max(space_dim[1:3]) * 7
    fig, ax = plt.subplots(layout = 'constrained', figsize = size)
    contorf = plt.contourf(output)
    contor = ax.contour(contorf, levels = contorf.levels[::2], colors='r')

    cbar_ticks = np.linspace(np.min(output), np.max(output), 20)
    cbar = fig.colorbar(contorf, ticks=cbar_ticks)
    cbar.ax.set_ylabel('Neutron Brightness')
    cbar.add_lines(contor)
    plt.clim(0, 5e31)

    plt.ylabel(f"{plot_axis[1]} [pix]")
    plt.xlabel(f"{plot_axis[0]} [pix]")
    plt.title(f"Raytracing output, subject:{plot_name}, \u03BB:{wavelength}\u212B, "
              f"P0:{initial_polarization*100}%, axis:{axis}")

    if plot_name == '' or plot_name is None:
        IMG_NAME = 'Result'
    else:
        IMG_NAME = f'{plot_name} {wavelength}A {int(initial_polarization*100)}P'

    plt.savefig(f"plots/{IMG_NAME} {axis}.png")
    print(f"{datetime.now()} - Plot saved as {IMG_NAME} {axis}.png")
        