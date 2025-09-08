# DL-RASER
**Correcting RASER MRI artifacts with Deep Learning** \
**Reference**: [10.1126/sciadv.abp8483](https://www.science.org/doi/10.1126/sciadv.abp8483) * supplementary materials heavily referenced

<hr>  

## Files
`RASER MRI sim 5.py` - created by Alon Greenbaum to solve equations S5-S8 in the supplementary materials linked above

`RASER-simulation.ipynb` - jupyter notebook that takes Alon's code and provides further explanation on functions and variables

`simulate-RASER.py` - Python file with edited script from `RASER-simulation.ipynb`

`simulation_log.txt` - first simulation log generated (was saved in the wrong folder), kept for reference

`requirements.txt` - required packages to run any of the above files, see below for instructions on installing

`2-mode-raser` directory - simulation of 2-mode system, developed by Chris Nelson and Seth Dilday

## Requirements
If you'd like to batch install the packages in the `requirements.txt` file, do the following: \
*If using Conda*, run the following in your command line: \
```conda install requirements.txt```

*Otherwise*, run the following in your command line: \
```pip install requirements.txt```