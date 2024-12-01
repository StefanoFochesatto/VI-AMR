# VI-AMR

This repository contains algorithms for adaptive mesh refinement with a goal of targeting refinement to the free boundary.  These codes support the research presented in my thesis.

There are two methods, called Unstructured Dilation Operator (UDO) and Varable Coefficient Elliptic Smoothing (VCES).  These are methods of the class `VIAMR` which is implemented in `src/viamr.py`.

## Dependencies

To get started, install firedrake along with the netgen/ngsolve integration.  Follow the instructions at [Firedrake download page](https://www.firedrakeproject.org/firedrake/download.html).

On an new firedrake install, netgen/ngsolve integration is added by running the firedrake-install script with the `--netgen` flag.

```
python3 firedrake-install --netgen
```

An existing firedrake install can be updated to include netgen/ngsolve integration by
```
(firedrake) python3 firedrake-update --netgen
```
from  running the firedrake-update script in `firedrake/bin/`.  Make sure the firedrake virtual environment is active before doing so.

FIXME I THINK VERSIONS IN requirements.txt UNDESIRABLE, AND mpi4py SHOULD BE REMOVED  Certain tests have additional dependencies.  These can be installed using
```
pip install -r requirements.txt
```

## Installation

Install editable with pip:
```
pip install -e .
```

## Usage

A first example compares 3 levels of refinement with UDO and VCES, using default parameters, on an obstacle problem with known exact solution.  In `examples/` do
```
python3 sphere.py
```
Then view the fields in `result_udo.pvd` and `result_vces.pvd` using [Paraview]().  For this example these files contain the obstacle `psi`, the solution `u`, and the numerical error `|u - uexact|`.

Clean up the results files with
```
make clean
```

FIXME from here

#### Numerical Results

##### Convergence Results

Recreating the convergence results for VCES can be done by navigating to _NumericalResults/ConvergenceResults/VCES_ and running

```
(firedrake) python3 Results.py
```

Results for UDO can be recreated analogously. Parameters like number of iterations, thresholding, and neighborhood depth can be set for each experiment using global variables in the corresponding `RunSolution.py` script located in the same directory.

##### Load Balancing Example

To recreate the inactive set load balancing example navigate to _NumericalResults/LoadBalancingExample_ and run the following

```
(firedrake) python3 GenerateSphereMesh.py --iter 5 --trih 0.45 --thresh 0.2 0.8 --proc 5
```

For this script, the experiment parameters like VCES iterations, initial mesh triangle height, thresholding parameters and number of processes can be adjusted using kwargs.

##### Parameter Exploration

To recreate the parameter exploration results navigate to _NumericalResults/ParameterExploration/VCES_ and run the following command

```
(firedrake) python3 Results.py
```
