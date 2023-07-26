"""This script takes the result of the ray tracing simulation,
    and plots the intensity of the neutron beam as a function of
    the initial polarization of the beam.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat

# Define the path to the data file
DATA_PATH = "Data_Folder/"
DATA_FILE = "Background.txt"

# Load and read the data
instance_name = DATA_FILE.split(".")[0]

with open(DATA_PATH + DATA_FILE, 'r') as f:
    data = f.readlines()

wavelength = np.array([])
P0 = np.array([])
intensity = np.array([])

for line in data:
    wavelength_temp, P0_temp, intensity_temp = line.split(" ")

    wavelength = wavelength_temp[3:-4]
    P0 = np.append(P0, float(P0_temp[3:-2]))
    intensity = np.append(intensity, float(intensity_temp.split(":")[1]))

# Make a fit to verify linearity
def power_series(x, a, b, c, d, e, f):
    """A 5-terms power series function"""
    output = a + b * x + c * x**2 + d * x**3 + e * x**4 + f * x**5
    return output

popt, pcov = curve_fit(power_series, P0, intensity)

fit_result = power_series(P0, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
residual = intensity - fit_result
chisq = sum((intensity - fit_result)**2 - fit_result)

# Plot the data
fig, ax = plt.subplots(2, 1, figsize=(9, 7))

# ax.plot(P0, intensity)
ax[0].scatter(P0, intensity, color = 'blue', label = "Data")
ax[0].plot(P0, fit_result, color = 'red', linestyle = '--', label = "Fit")
ax[0].set_ylabel("Intensity", fontsize = 15)
ax[0].set_xlim(1 + max(P0), min(P0) - 1)
ax[0].legend(fontsize = 15)
ax[0].grid()

ax[1].scatter(P0, residual, color = 'blue', label = "Residual")
ax[1].set_ylabel("Residual", fontsize = 15)
ax[1].grid()
ax[1].hlines(0, min(P0) - 1, 1 + max(P0), color = 'black', label = "Zero")
ax[1].legend(loc = 'upper left', fontsize = 15)
ax[1].set_xlim(1 + max(P0), min(P0) - 1)
ax[1].set_ylim(min(residual) * 1.5, max(residual) * 1.5)

fig.suptitle(f"Intensity vs initial polarization, {instance_name}, \u03BB={wavelength}\u212B", fontsize = 15)
fig.supxlabel("Initial Polarization [%]", fontsize = 15)
fig.tight_layout()

# plt.show()

print(f"The fit result is: y = {popt[0]} + {popt[1]}x + {popt[2]}x^2 + "
      f"{popt[3]}x^3 + {popt[4]}x^4 + {popt[5]}x^5")
print(f"The chi squared is: {chisq}")

fig.savefig(f"plots/Intensity_vs_P0 for {instance_name}.png")

half_i0 = ufloat(popt[0], (pcov[0][0])**0.5)
first_term = ufloat(popt[1], (pcov[1][1])**0.5)

dP = first_term / (half_i0 * P0)
