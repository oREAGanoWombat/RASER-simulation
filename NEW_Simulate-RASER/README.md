# Simulating RASER Dynamics
**Reference**: [10.1126/sciadv.abp8483](https://www.science.org/doi/10.1126/sciadv.abp8483) * supplementary materials heavily referenced

<hr>

## Files
`requirements.txt` - required packages to run any of the above files, see below for instructions on installing

`config.yml` - configuration file used in all scripts

## Requirements
If you'd like to batch install the packages in the `requirements.txt` file, do the following: \
*If using Conda*, run the following in your command line: \
```conda install requirements.txt```

*Otherwise*, run the following in your command line: \
```pip install requirements.txt```

## Configuration
Below is a list of parameters and settings that can be changed in `config.yml.` 
### Utility Variables
#### output_directory_root
- *variable must be encased in "quotation marks"*
- 'auto': setting this parameter to "auto" saves simulation outputs in a subdirectory named "Sim_RASER_Output" inside the root directory where the script is located
- setting this parameter to a specific path encased in quotation marks saves simulation outputs in the directory defined by the path
#### inversion_map_mode
- *variable must be encased in "quotation marks"*
- 'square': initial population inversion map is generated as a square of `shape=(n_modes, n_modes)` with uniform population inversion `d0_total`
- 'circle': initial population inversion map is generated as a circle with uniform population inversion
- 'ellipse': initial population inversion map is generated as an ellipse with uniform population inversion
- 'random': initial population inversion map is generated as random blobs with varying intensity

<hr>

## Simulate RASER Dynamics
Core simulation function `simulate_raser_dynamics()` \
This function models the time evolution of multiple interacting RASER (Radio-frequency Amplification by Stimulated Emission of Radiation) modes based on a set of coupled non-linear ordinary differential equations (ODEs), as described in the supplementary material (equations S5-S9). It takes an initial 1D population inversion profile and simulates how the population inversion, mode amplitudes, and phases change over time, ultimately calculating the total measurable output signal.
### Parameters
- `initial_population_inversion` (numpy.ndarray): a 1D NumPy array representing the initial population inversion $d_{\mu}(0)$ for each RASER mode/slice; this array is typically a 1D projection obtained from the 2D image
- `T1` (float, default: 5.0): the longitudinal relaxation time $T_1$ in seconds; this parameter governs how quickly the population inversion recovers to its equilibrium state (or decays if no pumping is present)
- `T2` (float, default: 0.7): the effective transverse relation time $T_2^*$ in seconds; this parameter describes the decay of the transverse magnetization (and thus the amplitude for each mode)
- `coupling_beta` (float, default: 1.0): the coupling parameter $\beta$ that dictates the strength of the interaction between the RASER modes and the overall gain; it's derived from physical constant and resonator properties
- `center_freq_hz` (float, default: 50.0): the center frequency $\nu_0$ of the RASER resonator in Hertz; this is the central frequency around which the individual RASER modes are distributed
- `gain_bandwidth_hz` (float, default: 10.0): the total bandwidth $\Delta$ of the imaging domain in Hertz, which corresponds to the frequency range covered by the magnetic field gradient
- `mode_spacing_hz` (float, default: 0.2): the frequency separation $\delta \nu$ between adjacent RASER modes in Hertz
- `sim_duration` (float, default: 2.0): the total duration of the simulation in seconds
- `points_per_sec` (int, default: 2000): the number of data points to generate per second of simulation time for the output signal; this affects the temporal resolution of the `output_signal`

### Returns: `dict`
A dictionary containing the results of the simulation.
- `time` (numpy.ndarray): a 1D array of time points (in seconds) at which the output signal was sampled
- `n_modes` (int): the number of RASER modes/slices ($N$) simulated
- `initial_inversion` (numpy.ndarray): the 1D array of initial population inversion values for each mode, as provided to the function
- `final_inversion` (numpy.ndarray): a 1D array of the population inversion values for each mode at the end of the simulation
- `final_amplitude` (numpy.ndarray): a 1D array of the amplitude values ($A_\mu$) for each mode at the end of the simulation
- `output_signal` (numpy.ndarray): a 1D array representing the total simulated RASER signal ($Sig(t)$) (see Eq. S9) over the `sim_duration`