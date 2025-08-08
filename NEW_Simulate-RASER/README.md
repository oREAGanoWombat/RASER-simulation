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
Below is a list of parameters and settings that can be changed in `config.yml.` \
**Setup** \
output_directory_root
- 'auto': setting this parameter to "auto" saves simulation outputs in a subdirectory named "Sim_RASER_Output" inside the root directory where the script is located
- setting this parameter to a specific path encased in quotation marks saves simulation outputs in the directory defined by the path \
inversion_map_mode
- 'square': initial population inversion map is generated as a square of `shape=(n_modes, n_modes)` with uniform population inversion `d0_total`
- 'circle'
- 'ellipse'
- 'random'