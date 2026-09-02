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


# Results Discussion

We should have a discussion about figures and make a concrete list.

Here is an explanation of the current codes for figure generation in `NumericalResults/ConvergenceResults`

 - `GenerateSolution.py`
 This code generates a high quality solution of the sphere obstacle problem to compare against. The initial mesh is constructed using netgen's [Constructive Solid Geometry](https://docu.ngsolve.org/latest/i-tutorials/unit-4.1.2-csg2d/csg2d.html) to define the boundary and the free boundary. The `fbres` controls the resolution which the quadratic splines are sampled. The `fbres` parameter and the `maxh` parameter need to be balanced for the meshing to succeed. Then the code alternates between VCES and Uniform refinement for 10 iterations and the mesh and solution are checkpointed. 

```
rect = Solid2d([
    (-2, -2),
    (-2, 2),
    (2, 2),
    (2, -2),
], mat="rect", bc="boundary")

r = 0.697965148223374
fbres = .001
circle = Solid2d([
    (0, -r),
    EI((r,  -r), maxh=fbres),  # control point for quadratic spline
    (r, 0),
    EI((r,  r), maxh=fbres),  # spline with maxh
    (0, r),
    EI((-r,  r), maxh=fbres),
    (-r, 0),
    EI((-r, -r), maxh=fbres),  # spline with bc
])

geo.Add(circle)
geo.Add(rect)
ngmsh = geo.GenerateMesh(maxh=.01)
labels = [i+1 for i,
          name in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ["boundary"]]
ExactMesh = Mesh(ngmsh)

```

 - `Convergence.py` 
This code runs a convergence analysis given the following kwargs: 
--initTriHeight: Initial mesh triangle height
--refinement: Whether the refinement strategy is: Uniform, AMR, or Hybrid (the hybrid strategy is hard coded in here and involves non trivial decision of when to switch between uniform and AMR; abstracting this will be necessary if we want to play with just inactive set refinement or DWR over inactive set).
--maxIterations: The maximum iterations to refine. 
--amrMethod: When we do AMR we need to specify VCES or UDO (we will need to add another option for metric based adaption)

So we run the analysis and the code spits out CSVs whose titles identify the refinement strategy and AMR method. The header for the CSVs looks like; L2, Jaccard, Hausdorff, Elements, Vertices, Sizes (hmin, hmax).

TODO: Get rid of meshHistory it is not necessary and it is slow and bad. Add timing data. Add metric based adaptation, possibly abstract Hybrid scheme implementation.

TODO: Add H1 norm, to computation

- `GeneratePlots.py`
This is the driver code for `Convergence.py`, so when we run `GeneratePlots.py --runconvergence` we use subprocess to run `Convergence.py` with every combination of --refinement and --amrMethod. Without the `--runconvergence` flag we assume the CSVs have been generated and they get read into a dataframe and the `getPlot()` function will plot any two columns against each other for every --amrMethod.

TODO: Add functionality for timing data and metric amr. 

TLDR: Run `GenerateSolution.py` for the "exact solution" . Then `GeneratePlots.py --runconvergence` once, to get the data use or modify `getPlot()` to get different plots.  


 
 Here is a potential list @bueler : 
For all these methods we plot Uniform, AMR only, and Hybrid (so three lines)
VCES | UDO | Metric

x vs y
 - Elements vs L2
 - Elements vs Jaccard
 - Elements vs Hausdorff
^ These are essentially in my thesis
Time is not clear to me. Should we time from the end of the solve to the construction of the new mesh, so just the refinement step? 

 - Time vs L2
 - Time vs Jaccard
 - Time vs Hausdorff['vcesBR', 'udoBR', 'metricIsoHess'] c
 - dof vs solve time