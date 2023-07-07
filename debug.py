"""Main file for the project."""
import numpy as np
from matplotlib import pyplot as plt

# Configuration
DATA_PATH = "E:/Documents and stuff/School_Stuff/_CSNS/PNI/COMSOL6.0(64bit)/Simulations/"
# DATA_FILE = "Horseshoe.txt"
DATA_FILE = "bar_magnet.txt"

AXIS = "x"

PLOT_FIELD_FIRST = True # If true, generates a plot of the field before raytracing
SOURCE_PROFILE = "gaussian" # Defines a source profile (use an imported function here)


def read_data(data_path, filename, facing_dir = "x"):
    """Reads data from COMSOL output file.
    
    Args:
        data_path (str): Path to the data file.
        filename (str): Name of the data file.
        facing_dir (str): Direction in which the neutron is travelling. Defaults to "x".

    Returns:
        field (np.ndarray): Magnetic field data.
        field_loc (np.ndarray): Location of the magnetic field data.
    """
    axis = {"x": [0, 1, 2], "y": [1, 0, 2], "z": [2, 0, 1]}
    axis_loc = np.array(axis[facing_dir])
    axis_field = axis_loc + 3

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

    temp1, temp2, temp3 = axis_field
    field_input = np.array([data[temp3], data[temp2], data[temp1]])

    temp1, temp2, temp3 = axis_loc
    field_loc = np.array([data[temp3], data[temp2], data[temp1]])

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

# Execution code
if __name__ == "__main__":
    # Identify input data type
    if DATA_FILE.rsplit('.', maxsplit=1)[-1] == 'txt':
        field, _ = read_data(DATA_PATH, DATA_FILE, facing_dir = "x")
        field_loc = np.indices(np.shape(field)[:-1])
    elif DATA_FILE.rsplit('.', maxsplit=1)[-1] == 'npz':
        data = np.load(DATA_PATH + "/" + DATA_FILE)
        field = data["field"]
        field_loc = data["field_loc"]

    # Trim the data
    # field = field[1:-1, 1:-1, 1:-1, :]
    # field_loc = field_loc[:, 1:-1, 1:-1, 1:-1]

    # Rotate the field
    axis_rot = {"x": [0, 1, 2], "y": [1, 0, 2], "z": [2, 0, 1]}

    field_T = np.transpose(field, (3, 0, 1, 2))
    field_T = field_T[axis_rot[AXIS]]
    field = np.transpose(field_T, (1, 2, 3, 0))

    field_loc = field_loc[axis_rot[AXIS]]

    space_dim = np.shape(field)
    space_dim = np.array(space_dim[:-1])-2

    # Plot the input data
    if PLOT_FIELD_FIRST:
        ax = plt.figure().add_subplot(projection='3d')
        field_temp = np.copy(field_T)
        x, y, z = field_loc
        ax.quiver(x, y, z,
                  field_temp[0], field_temp[1], field_temp[2],
                  length = 1, normalize = True)
        # plt.xlim(0, space_dim[0])
        # plt.ylim(0, space_dim[1])
        # ax.set_zlim(0, space_dim[2])
        plt.show()