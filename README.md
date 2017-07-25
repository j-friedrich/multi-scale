# Multi-scale approaches for high-speed imaging and analysis of large neural populations

This code accompanies the paper [Multi-scale approaches for high-speed imaging and analysis of large neural populations](https://doi.org/10.1101/091132) [bioRxiv, 2016]


### Requirements
The scripts were tested on Linux and MacOS with a typical numerical/scientific Python 2.7 installation, e.g. using Anaconda or Canopy.

The scripts make use of [CaImAn](https://github.com/simonsfoundation/CaImAn) (formerly Constrained_NMF), hence all its dependencies have to be met. 
To avoid issues due to newer versions of CaImAn, we recommend to clone/download it from [this branch](https://github.com/j-friedrich/Constrained_NMF/tree/multi-scale_paper), which further includes minor extensions to produce Fig 8. 
Please make sure to add the package Constrained_NMF to your $PYTHONPATH, so that the scripts in this repo find it.


### Execution
The scripts to produce the figures and table have names obvious from the bioRxiv paper. 
They can be run with `python fig[1-8].py`. 
