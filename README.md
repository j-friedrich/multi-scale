# Multi-scale approaches for high-speed imaging and analysis of large neural populations

This code accompanies the paper [Multi-scale approaches for high-speed imaging and analysis of large neural populations](https://doi.org/10.1371/journal.pcbi.1005685) [PLoS Comput Biol. 2017; 13(8):e1005685.]


### Requirements
The scripts were tested on Linux and MacOS with a typical numerical/scientific Python 2.7 installation, e.g. using Anaconda or Canopy.

The scripts make use of [CaImAn](https://github.com/simonsfoundation/CaImAn) (formerly Constrained_NMF), hence all its dependencies have to be met. 
To avoid issues due to newer versions of CaImAn, we recommend to clone/download it from [this branch](https://github.com/j-friedrich/CaImAn/tree/multi-scale_paper), which further includes minor extensions to produce Fig 8. 
Please make sure to add the package Constrained_NMF to your $PYTHONPATH, so that the scripts in this repo find it.


### Execution
The scripts to produce the figures and table have names obvious from the PLoS Comput Biol paper. 
They can be run with `python table1.py` and `python fig[1-8].py` to show the figures during code execution. If a (sub)directory name is provided as argument, e.g. `python fig1.py fig`, figures are saved in the directory, e.g. `fig`, if it exists. For the two-photon data, fig[4-8], you need to execute `python run_2P.py` first, which could take few hours and saves the results in the subdirectory `results`.
