"""Main file for the project."""
import multiprocessing as mp
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import config
import config_fns
import ray_tracing_fns
import os

# # Put the input file path and file names here
DATA_PATH = "E:/Documents and stuff/School_Stuff/_CSNS/PNI/COMSOL6.0(64bit)/Simulations/"
# DATA_FILE = "Horseshoe.txt"
# DATA_FILE = "bar_magnet.txt"
DATA_FILE = "empty.txt"

# # Defines a source profile
# # To use a uniform source profile, set SOURCE_PROFILE to None
# # To use a gaussian source profile, set SOURCE_PROFILE to "gaussian"
# # To use a custom source profile, set SOURCE_PROFILE to a 2D intensity map (np.array)
SOURCE_PROFILE = "gaussian"

# # Specify in which axis is the neutron beam projected
AXIS = "y" # Axis to be raytraced along, must be "x", "y", or "z"

# # Physisc settings
INITIAL_POLARIZATION = 0.99 # Initial polarization rate of the neutron
WAVELENGTH = 2

# # Misc settings
PLOT_NAME = 'sim'
PLOT_FIELD_FIRST = False # If true, generates a plot of the field before raytracing
SHOW_PROGRESS = True # If true, prints the progress


# Execution code
if __name__ == "__main__":
    # Identify input data type
    if DATA_FILE.rsplit('.', maxsplit=1)[-1] == 'txt':
        field, _ = config_fns.read_data(DATA_PATH, DATA_FILE)
        field_loc = np.indices(np.shape(field)[:-1])
    elif DATA_FILE.rsplit('.', maxsplit=1)[-1] == 'npz':
        data = np.load(DATA_PATH + "/" + DATA_FILE)
        field = data["field"]
        field_loc = data["field_loc"]

    field = np.transpose(field, (3, 0, 1, 2))

    # Rotate the field (40, 32, 24, 3) (z, y, x, 3)
    axis_rot = {"x": (3, 2, 1, 0), "y": (2, 3, 1, 0), "z": (1, 3, 2, 0)}

    field_loc = np.transpose(field_loc, axis_rot[AXIS])
    field = np.transpose(field, axis_rot[AXIS])

    space_dim = np.shape(field)[0:3]
    print(f"The size of the space is {space_dim}")
    space_dim = np.array(space_dim) + [space_dim[0]//10, space_dim[1]//10, space_dim[2]//10]

    field_loc_T = np.transpose(field_loc, (3, 0, 1, 2))
    field_T = np.transpose(field, (3, 0, 1, 2))

    # Plot the input data
    if PLOT_FIELD_FIRST:
        ax = plt.figure().add_subplot(projection='3d')
        x, y, z = field_loc_T
        ax.quiver(x, y, z,
                  field_T[2], field_T[1], field_T[0],
                  length = 1, normalize = True)
        plt.show()

    if SOURCE_PROFILE is None:
        source = np.ones((space_dim[1], space_dim[2]))
        print("No source profile specified, using a uniform source profile.")
    elif SOURCE_PROFILE == "gaussian":
        source = config_fns.gaussian_2d(space_dim,
                                        config.SOURCE_STD,
                                        config.SOURCE_NORM)
    else: 
        pass

    # Do raytracing
    start_time = datetime.now()
    print(f"Starting raytracing at {str(start_time)}, along axis {AXIS}.")

    output = np.zeros(space_dim[1:3])

    ROW_TIME = 0
    tot_row = np.shape(output)[0]
    # NOTE: I might be iterating through the dimensions in the wrong order
    with mp.Pool(processes = os.cpu_count()) as pool:
        for i, row in enumerate(output):
            row_start = datetime.now()
            progress = round(i/tot_row * 100, 2)
            if SHOW_PROGRESS:
                print(f"Now processing row {i}/{tot_row}, progress {progress}"
                      f"%. Last row used {ROW_TIME}. "
                      f"Total time elapsed: {datetime.now() - start_time}. "
                      f"Estimated time remaining: {tot_row * ROW_TIME}. ")
            for j, col in enumerate(row):
                i_pix = pool.apply_async(ray_tracing_fns.ray_tracing_sim,
                                         args=(field, (i, j), source))
                i_pix = i_pix.get()

                output[i][j] = i_pix

            ROW_TIME = datetime.now() - row_start

    # Normalize the output

    completion = datetime.now()
    print(f"{completion} - Processing complete."
          f"Total time taken: {completion - start_time}")

    PLOT_AXIS = "zyx".replace(AXIS, '')

    # Make the plot
    size = (space_dim[2] + 1.5, space_dim[1])/max(space_dim[1:3]) * 7
    fig, ax = plt.subplots(layout = 'constrained', figsize = size)
    contorf = plt.contourf(output)
    contor = ax.contour(contorf, levels = contorf.levels[::2], colors='r')

    cbar_ticks = np.linspace(np.min(output), np.max(output), 20)
    cbar = fig.colorbar(contorf, ticks=cbar_ticks)
    cbar.ax.set_ylabel('Neutron Brightness')
    cbar.add_lines(contor)

    plt.ylabel(f"{PLOT_AXIS[1]} [pix]")
    plt.xlabel(f"{PLOT_AXIS[0]} [pix]")
    plt.title(f"Output of raytracing simulation, along {AXIS} axis")

    if PLOT_NAME == '' or PLOT_NAME is None:
        PLOT_NAME = 'result'
    else:
        PLOT_NAME = f'{PLOT_NAME} {WAVELENGTH}A {int(INITIAL_POLARIZATION*100)}P'

    plt.savefig(f"plots/{PLOT_NAME} {AXIS}.png")
    print(f"{datetime.now()} - Plot saved as {PLOT_NAME} {AXIS}.png")
    plt.show()
