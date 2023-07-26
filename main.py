"""Main file for the project."""
import numpy as np
import config_fns as cf

# # Put the input file path and file names here
DATA_PATH = "E:/Documents and stuff/School_Stuff/_CSNS/PNI/COMSOL6.0(64bit)/Simulations/"
# DATA_FILE = "Horseshoe.txt"
DATA_FILE = "bar_magnet.txt"
# DATA_FILE = "empty.txt"

# # Defines a source profile
# # To use a uniform source profile, set SOURCE_PROFILE to None
# # To use a gaussian source profile, set SOURCE_PROFILE to "gaussian"
# # To use a custom source profile, set SOURCE_PROFILE to a 2D intensity map (np.array)
SOURCE_PROFILE = "gaussian"

# # Specify in which axis is the neutron beam projected
AXIS = "x" # Axis to be raytraced along, must be "x", "y", or "z"

# # Physisc settings
INITIAL_POLARIZATION = [0.99] # Initial polarization rate of the neutron in decimals
WAVELENGTH = np.arange(60, 150, 1) # Wavelength of the neutron in Angstroms

# # Misc settings
PLOT_NAME = 'Simulation'
PLOT_FIELD_FIRST = False # If true, generates a plot of the field before raytracing
SHOW_PROGRESS = True # If true, prints the progress
MAKE_PLOT = False


# # Execution code
if __name__ == "__main__":
    for i in INITIAL_POLARIZATION:
        for j in WAVELENGTH:
            wl = round(j, 1)
            print(f"Starting simulation for \u03BB={wl}\u212B, "
                  f"P0 = {i*100}%.")
            cf.master_execution(data_path=DATA_PATH,
                                data_file=DATA_FILE,
                                source_profile=SOURCE_PROFILE,
                                axis=AXIS,
                                initial_polarization=i,
                                wavelength=wl,
                                plot_name=PLOT_NAME,
                                plot_field_first=PLOT_FIELD_FIRST,
                                show_progress=SHOW_PROGRESS,
                                make_plot=MAKE_PLOT)
