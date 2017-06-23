# mpc_comparison_rpic2017

Support material for "MPC for nonlinear systems: a comparative review of discretization methods".

In order to run these examples you will need [CasADi](http://www.casadi.org) version >= 3.1.0 to be installed on your system and (optionally) the [HSL MA27](http://www.hsl.rl.ac.uk/ipopt/) linear solver. You can find CasADi installation instructions [here](https://github.com/casadi/casadi/wiki/InstallationInstructions). Once you have CasADi working, you can follow HSL installation instructions [here](https://github.com/casadi/casadi/wiki/Obtaining-HSL). If you don't plan to install the HSL solvers, just comment out the line that says `"linear_solver" : "ma27"` in each of the examples.

## Creating a Python virtual environment to run the provided examples

If you have `virtualenv` installed, you can configure an isolated Python environment to run the coding examples. This repository provides a `virtualenv_requirements.txt` file to facilitate the installation of the required dependencies.

First, you need to create a directory to contain your virtual environments and then create one for this project:
```
~$ mkdir virtualenvs
~$ cd virtualenvs
~/virtualenvs$ virtualenv mpc_rpic2017
```
Now, the virtual environment needs to be activated:
```
~/virtualenvs$ source mpc_rpic2017/bin/activate
```
The name of the current virtual environment will now appear on the left of the prompt to let you know that it’s active. From now on, any package that you install using pip will be placed in the `mpc_rpic2017` folder, isolated from the global Python installation.

Download the file `virtualenv_requirements.txt` and place it in the current folder. Then, we can install the required dependencies
```
~/virtualenvs$ pip install -r virtualenv_requirements.txt
```
and test the provided code. For example, to test the script `vdp_comparison_collocation.py`, you can run
```
$ python vdp_comparison_collocation.py
```

Once you are done working in the virtual environment for the moment, you can deactivate it:
```
$ deactivate
```
Now running `python` will just use the system’s default Python interpreter, which is not modified by anything done while being inside the virtual environment.

To delete a virtual environment, just delete the corresponding folder. (In this case, it would be `rm -r mpc_rpic2017`).
