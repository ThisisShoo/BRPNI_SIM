![alt text](https://github.com/ThisisShoo/PNI_Sim/blob/main/Misc/animation.gif "Logo Title Text 1")

# BRPNI_SIM

This neutron Larmor precession simulator is compatible with a diffusive neutron source with a given intensity profile. It was designed to verify theories regarding Polarized Neutron Imaging (PNI). The simulator is written in Python 3.11 and uses the following packages:

* numpy
* matplotlib
* multiprocessing
* datetime

This package also contains two post-simulation analysis scripts, which are also written in Python. `fit_background.py` is used to fit the background of the simulated data, and `fit_data.py` is used to fit the simulated data to a Gaussian function. Both scripts use the following packages:

* numpy
* matplotlib
* scipy

Please find the presentation slides for this project [here](https://github.com/ThisisShoo/PNI_Sim/blob/main/Misc/Algorithm Presentation.pptx)

## Installation

To install the simulator, simply clone the repository. 

## Usage

All user-facing variables are defined in the beginning of main.py, and they can be modified to suit the user's needs. See the comments in main.py for more information on each variable.
To run the simulator, simply run main.py. The output will be saved in the `plots` folder. 

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements

This project was developed by Shuhan Zheng (University of Toronto, GitHub: https://github.com/ThisisShoo) for his 2023 summer research project at China Spallation Neutron Source (CSNS), under the guidance of Dr. Tianhao Wang, and many students at Polarized Neutron Group at CSNS. It would not have been possible without their help.
