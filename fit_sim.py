"""This script takes the result of the ray tracing simulation,
    and plots the intensity of the neutron beam as a function of
    the overall wavelength of the beam.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Define the path to the data file
DATA_PATH = "Data_Folder/"
DATA_FILE = "Simulation.txt"

# Load and read the data
instance_name = DATA_FILE.split(".")[0]

with open(DATA_PATH + DATA_FILE, 'r') as f:
    data = f.readlines()

wavelength = np.array([])
P0 = np.array([])
intensity = np.array([])

for line in data:
    wavelength_temp, P0_temp, intensity_temp = line.split(" ")

    wavelength = np.append(wavelength, float(wavelength_temp[3:-4]))
    P0 = float(P0_temp[3:-2])
    intensity = np.append(intensity, float(intensity_temp.split(":")[1]))

# Make a fit to verify linearity
def power_series(x, a, b, c, d, e, f_, g):
    """A 5-terms power series function"""
    output = a + b * x + c * x**2 + d * x**3 + e * x**4 + f_ * x**5 + g * x**6
    return output

popt, pcov = curve_fit(power_series, wavelength, intensity, p0=[1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30])

fit_result = power_series(wavelength, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
residual = fit_result - intensity
chisq = sum((intensity - fit_result)**2 - fit_result)

# Plot the data
fig, ax = plt.subplots(2, 1, figsize=(9, 7))

# ax.plot(P0, intensity)
ax[0].scatter(wavelength, intensity, color = 'blue', label = "Data")
ax[0].plot(wavelength, fit_result, color = 'red', linestyle = '--', label = "Fit")
ax[0].set_ylabel("Intensity")
ax[0].set_title(f"Intensity vs wavelength, {instance_name}, P0={P0}%")
ax[0].set_xlim(min(wavelength) - 1, 1 + max(wavelength))

ax[0].legend(loc = 'upper left')
ax[0].grid()

ax[1].scatter(wavelength, residual, color = 'blue', label = "Residual")
ax[1].set_ylabel("Residual")
ax[1].grid()
ax[1].hlines(0, min(wavelength) - 1, 1 + max(wavelength), color = 'black', label = "Zero")
ax[1].legend(loc = 'upper left')
ax[1].set_xlim(min(wavelength) - 1, 1 + max(wavelength))
ax[1].set_ylim(min(residual) * 1.5, max(residual) * 1.5)

fig.supxlabel("Initial Polarization [%]")
fig.tight_layout()

# plt.show()

print(f"The fit result is: y = {popt[0]} + {popt[1]}x + {popt[2]}x^2 + "
      f"{popt[3]}x^3 + {popt[4]}x^4 + {popt[5]}x^5 + {popt[6]}x^6")
print(f"The chi squared is: {chisq}")

fig.savefig(f"plots/Intensity_vs_wavelength for {instance_name}.png")
