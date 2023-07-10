"""Main file for the project."""
import multiprocessing as mp
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import config
import config_fns
import ray_tracing_fns
import os

# Configuration
DATA_PATH = "E:/Documents and stuff/School_Stuff/_CSNS/PNI/COMSOL6.0(64bit)/Simulations/"
# DATA_FILE = "Horseshoe.txt"
DATA_FILE = "bar_magnet.txt"

AXIS = "y" # Axis to be raytraced along, must be "x", "y", or "z"

PLOT_FIELD_FIRST = False # If true, generates a plot of the field before raytracing
SHOW_PROGRESS = False # If true, prints the progress
SOURCE_PROFILE = "gaussian" # Defines a source profile (use an imported function here)

# Execution code
if __name__ == "__main__":
    # Identify input data type
    if DATA_FILE.rsplit('.', maxsplit=1)[-1] == 'txt':
        field, _ = config_fns.read_data(DATA_PATH, DATA_FILE, facing_dir = "x")
        field_loc = np.indices(np.shape(field)[:-1])
    elif DATA_FILE.rsplit('.', maxsplit=1)[-1] == 'npz':
        data = np.load(DATA_PATH + "/" + DATA_FILE)
        field = data["field"]
        field_loc = data["field_loc"]

    field = np.transpose(field, (3, 0, 1, 2))

    # Rotate the field
    axis_rot = {"x": (3, 2, 1, 0), "y": (2, 3, 1, 0), "z": (1, 3, 2, 0)}
    
    field_loc = np.transpose(field_loc, axis_rot[AXIS])
    field = np.transpose(field, axis_rot[AXIS])

    print(np.shape(field_loc), np.shape(field))

    space_dim = np.shape(field)[1:]
    print(f"The size of the space is {space_dim[2]}")
    space_dim = np.array(space_dim)

    field_loc_T = np.transpose(field_loc, (3, 0, 1, 2))
    field_T = np.transpose(field, (3, 0, 1, 2))

    # Plot the input data
    if PLOT_FIELD_FIRST:
        ax = plt.figure().add_subplot(projection='3d')
        x, y, z = field_loc_T
        ax.quiver(x, y, z,
                  field_T[2], field_T[1], field_T[0],
                  length = 1, normalize = True)
        # plt.xlim(0, space_dim[0])
        # plt.ylim(0, space_dim[1])
        # ax.set_zlim(0, space_dim[2])
        plt.show()

    if SOURCE_PROFILE is None:
        source = np.ones((space_dim[1], space_dim[2]))
    elif SOURCE_PROFILE == "gaussian":
        source = config_fns.gaussian_2d(space_dim,
                                        config.SOURCE_STD,
                                        config.SOURCE_NORM)
    else: 
        raise ValueError("Invalid source profile.")

    # Do raytracing
    start_time = datetime.now()
    print("Starting raytracing at " + str(start_time))

    output = np.zeros(space_dim[0:2])
    print(np.shape(output))

    PIX_COUNT = 0
    PIX_TIME = 0
    tot_pix = np.prod(np.shape(output))
    with mp.Pool(processes = os.cpu_count()) as pool:
        for i, row in enumerate(output):
            for j, col in enumerate(row):
                pix_start = datetime.now()
                progress = round(PIX_COUNT/tot_pix * 100, 2)
                
                if SHOW_PROGRESS:
                    print(f"Now processing pixel ({i}, {j}), progress {progress}%. "
                      f"Last pixel used {PIX_TIME}. "
                      f"Total time elapsed: {datetime.now() - start_time}")
                i_pix = pool.apply_async(ray_tracing_fns.ray_tracing_sim,
                                         args=(field, (i, j), source))
                i_pix = i_pix.get()

                output[i][j] = i_pix

                PIX_COUNT += 1
                PIX_TIME = datetime.now() - pix_start

    # Normalize the output

    completion = datetime.now()
    print(f"{completion} - Processing complete."
          f"Total time taken: {completion - start_time}")

    PLOT_AXIS = "xyz".replace(AXIS, '')

    # Make the plot
    fig, ax = plt.subplots(layout = 'constrained')
    contorf = plt.contourf(output)
    contor = ax.contour(contorf, levels = contorf.levels[::2], colors='r')

    cbar = fig.colorbar(contorf)
    cbar.ax.set_ylabel('Neutron Brightness')
    cbar.add_lines(contor)

    plt.xlabel(f"{PLOT_AXIS[0]} [pix]")
    plt.ylabel(f"{PLOT_AXIS[1]} [pix]")
    plt.title(f"Output of raytracing simulation, {AXIS} axis")
    plt.savefig(f"plots/result {AXIS}.png")

