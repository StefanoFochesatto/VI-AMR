# VI AMR

This repository contains the code supporting the research presented in my thesis. It serves as the primary working space for managing and maintaining all scripts and modules used throughout the research process.

## Structure

### Main Methods

- **UDO and VCES**: These directories contain the primary examples for the proposed methods outlined in the thesis. They serve as the core implementation of the Unstructured Dilation Operator (UDO) and Varable Coefficient Elliptic Smoothing (VCES).

### Numerical Results

- **Convergence Results**: Contains scripts for generating the various convergence results regarding these AMR methods found near the end of the thesis
- **Load Balancing Examples**: Includes the final result regarding the inactive set balancing behavior these methods exhibit.
- **Parameter Exploration**: Contains scripts used for exploring thresholding and neighborhood depth parameters for both VCES and UDO respectively.

### Additional Scripts

- **Firedrake Examples and VI Convergence Experiment**: These folders encompass the remaining scripts utilized in the thesis. They include supplementary experiments and examples developed during the research, including those related to the Firedrake platform and VI convergence studies.

## Usage

To get started, install firedrake along with the netgen/ngsolve integration instructions for which can be found in the [Firedrake download page](https://www.firedrakeproject.org/firedrake/download.html).

On an new firedrake install, the netgen/ngsolve integration is installed by running the firedrake-install script with the '--netgen' flag.

```
python3 firedrake-install --netgen
```

Alternatively, a firedrake install can be updated to include the netgen/ngsolve integration by running the firedrake-update script in _firedrake/bin/firedrake-update_. Make sure the firedrake virtual environment is active before doing so.

```
(firedrake) python3 firedrake-update --netgen
```

The rest of the dependencies can be installed using

```
pip install -r requirements.txt
```

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
