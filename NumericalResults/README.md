#### Numerical Results

##### Prerequisites

- pandas
- matplotlib
- shapely
- meshio

##### Convergence Results

Recreating the convergence results for VCES can be done by navigating to _ConvergenceResults/VCES_ and running `results.py`. Results for UDO can be recreated analogously. Parameters like number of iterations, thresholding, and neighborhood depth can be set for each experiment using global variables in the corresponding `RunSolution.py` script located in the same directory.

##### Load Balancing Example

To recreate the inactive set load balancing example navigate to _LoadBalancingExample_ and run the following

```
python3 GenerateSphereMesh.py --iter 5 --trih 0.45 --thresh 0.2 0.8 --proc 5
```

For this script, the experiment parameters like VCES iterations, initial mesh triangle height, thresholding parameters and number of processes can be adjusted using kwargs.

##### Parameter Exploration

To recreate the parameter exploration results navigate to _ParameterExploration/VCES_ and run `Results.py`.
