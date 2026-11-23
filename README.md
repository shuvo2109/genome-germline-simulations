# Required Python Version
This code is written using Python 3.11. Any latest version of Python should be able to run the code.

# About the Game
TO-DO: Quick summary of the paper

# About the Code
The repository holds files to run simulations and generate relevant results and figures for the "De-Darwinizing the catalytic workforce: the genome as the original germ line" manuscript by Shuvo, Burnetti, and Ratcliff. The included files are:
- `hmodel.py`: Stochastic protocell model and analytical helpers. Defines parameters and fitness, reproduction and simulation files.
- `make_figures.py`: Simulates the coordination problem and de-Darwinization through translation and multilevel selection. Generates `Fig. 2` and `Fig. 3` of the manuscript. Utilizes functions in `hmodel.py`.
- `fixed_points_stability.py`: Simulates de-Darwinization as a continuum. Generates `Fig. 4` of the manuscript.
- `kstar_invasion.py`: Simulates ceiling and invasion. Generates `Fig. 5` of the manuscript. Utilizes functions in `hmodel.py`.

## How to use
Run `python` followed by the desired `.py` files to generate results and figures (expect `hmodel.py`).
Modify `hmodel.py` for desired parameter values.
Set `print_results = False` in `fixed_points_stability.py` and `kstar_invasion.py` if you only want the figures without the numerical results.

## Required libraries
The simulation primarily uses the following libraries. Please ensure you have the following libraries.
- `numpy`
- `scipy`
- `matplotlib`

## Acknowledgement
This repository pertains to a manuscript originally written and simulated by WC Ratcliff and AJ Burnetti, which was later expanded by MFR Shuvo with the revisions supervised by WC Ratcliff. Claude (Anthropic) was used to help write and test the code.

## Contact
- Md Fahimur Rahman Shuvo, md.shuvo@ufl.edu, rahman.fahimur21@gmail.com
- Corresponding author: William C. Ratcliff, ratcliff@gatech.edu.
