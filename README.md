ISSF Case 7
===========

This repository contains Python scripts, simulation setup files, and simulation data files for use in Industry Simulation Software for Fusion (ISSF) case study 5: reliability analysis of monoblock failure.

Prerequisites
-------------

  - A local MOOSE installation (see [proteus](https://github.com/aurora-multiphysics/proteus) or [MOOSE](https://mooseframework.inl.gov/index.html))
  - Python version >=3.7
  - [pathlib](https://docs.python.org/3/library/pathlib.html)
  - [uuid](https://docs.python.org/3/library/uuid.html)
  - [pandas](https://pandas.pydata.org/)

Operation
---------

The MOOSE FEA software that this case study uses can only run in a Linux environment. Installation should either be on a full Linux system or on Windows Subsystem for Linux (WSL). The repository includes a Python script ```run_issf_7_from_windows.py``` that facilitates execution from Windows using WSL.

Execution of the scripts is either from ```run_issf_7.py``` (Linux) or ```run_issf_7_from_windows.py``` (Windows calling WSL). In either case all user input is mananged within the clearly marked section of the script. Parameter input and output can be achieved either by acting on variables within the script or via input and ouput JSON files.
