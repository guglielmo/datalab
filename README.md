# datalab
Some data analysis notebooks, mostly done at Openpolis Foundation

# launch
This is a JupyterLab instance. It may be launched with:
```
jupyter lab
```
after the virtual environment has been initialised.

To launch in the proper virtualenv and have a shell available:
```
tmuxinator
```

# dedicated kernels
Custom kernels are defined in ``~/Library/Jupyter/kernels``. This allows a user to select a kernel that has access to the selected django environment,
making working with data extractions a breeze.
``ipykernel`` must be installed in the virtualenv of the project we want to work in (povedu, opdm, ...)

