"""Configuration file for neutron tracing. Edit this file to change the parameters of the simulation."""
import numpy as np
# Maximum angle of the neutron's trajectory with respect to the z-axis
MAX_ANGLE = 30
GAMMA = 1.83247179 * 10**8 # Neutron gyromagnetic ratio in rad/s/T
space_dim = (20, 20, 20)

# Gaussian beam parameters
SOURCE_STD = (1, 1)
SOURCE_NORM = (0.5, 0.5)
