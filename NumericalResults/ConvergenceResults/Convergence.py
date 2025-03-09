# Import Firedrake and Netgen
from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from viamr.utility import SphereObstacleProblem
import time
import math
import argparse
import pandas as pd
import os


vcesHybridVertexCounts = [413,623,2417,3221,12737,15837, 113]
vcesAdaptVertexCounts = [194,436,842,1644,3216,6302, 113]

methodlist = ['vces', 'udo', 'metricIso', 'vcesUnif', 'udoUnif', 'metricIsoHess', 'vcesBR', 'udoBR', 'uniform']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run convergence script with specified parameters.")
    parser.add_argument('-t', '--initTriHeight', type=float,
                        default=0.45, help='Initial triangle height')
    parser.add_argument('-i', '--maxIterations', type=int,
                        default=7, help='Maximum number of iterations')
    parser.add_argument('-m', '--method', type=str, 
                        default='uniform', help='string indicating method')
    
    args = parser.parse_args()
    method = args.method

    problem_instance = SphereObstacleProblem(TriHeight=args.initTriHeight)
    amr_instance = VIAMR()
    mesh = problem_instance.setInitialMesh()
    u = None

    # for debugging purposes
    #os.chdir("/home/stefano/Desktop/VI-AMR/NumericalResults/ConvergenceResults")
    
    # Load in data about exact solution
    with CheckpointFile("ExactSolution.h5", 'r') as afile:
        # The default name for checkpointing a netgen mesh is not the same as a firedrake mesh # this might be fixed in new firedrake builds 
        exactMesh = afile.load_mesh('Default')
        exactU = afile.load_function(exactMesh, "ExactU")
    exactV = FunctionSpace(exactMesh, "CG", 1)

    exactPsi = problem_instance.getObstacle(exactV)
    exactElementIndicator = amr_instance.elemactive(exactU, exactPsi)
    _, exactFreeBoundaryEdges = amr_instance.freeboundarygraph(
        exactU, exactPsi)

    # init data list which will be written to csv
    data = []
    for i in range(args.maxIterations):
        # solution gets overwritten; never stored
        tic = time.perf_counter()
        u, lb = problem_instance.solveProblem(mesh=mesh, u=u)
        toc = time.perf_counter()
        
        # Get Mesh details
        nv, ne, hmin, hmax = amr_instance.meshsizes(mesh)

        # Compute Jaccard index
        solElementIndicator = amr_instance.elemactive(u, lb)
        JError = amr_instance.jaccard(
            solElementIndicator, exactElementIndicator)

        # Compute Hausdorff error
        _, solFreeBoundaryEdges = amr_instance.freeboundarygraph(u, lb)
        HError = amr_instance.hausdorff(
            solFreeBoundaryEdges, exactFreeBoundaryEdges)

        # Compute L2 error
        _, exactU = problem_instance.getBoundaryConditions(u.function_space())
        diffu = Function(u.function_space()).interpolate(u - exactU)
        L2Error = sqrt(assemble(dot(diffu, diffu) * dx))
        
        # Compute H1 error

        # FIXME: investigate this further, this is giving weird results.
        # Compute L2 Error (using conservative projection to finer mesh)
        # proju = Function(exactV).project(u)
        # L2Error = sqrt(
        #    assemble(dot((proju - exactU), (proju - exactU)) * dx(exactMesh)))
        
        
        # Big switch style statement for methods
        if method == methodlist[0]: # vces
            mtic = time.perf_counter()
            mark = amr_instance.vcesmark(mesh, u, lb, bracket=[.2, .8])
            mesh = mesh.refine_marked_elements(mark)
            mtoc = time.perf_counter()

            
        elif method == methodlist[1]: # udo 
            mtic = time.perf_counter()
            mark = amr_instance.udomarkParallel(mesh, u, lb, n=3)
            mesh = mesh.refine_marked_elements(mark)
            mtoc = time.perf_counter()

            
        elif method == methodlist[2]: # metric isotropic only
            mtic = time.perf_counter()
            amr_instance.setmetricparameters(target_complexity=vcesAdaptVertexCounts[i])# Corresponds to only freeboundary metric applied
            mesh = amr_instance.metricrefine(mesh, u, lb, weights=[0, 1])
            mtoc = time.perf_counter()

        
        elif method == methodlist[3]: # vces + uniform
            CG1, DG0 = amr_instance.spaces(mesh)
            mtic = time.perf_counter()
            h = max(mesh.cell_sizes.dat.data)
            ratio = HError/(h**2)
            switch = math.isclose(ratio, 1, rel_tol=.1)
            if HError < h**2 or switch: # uniform
                mark = Function(DG0).interpolate(Constant(1.0))
                mesh = mesh.refine_marked_elements(mark)
                
            else: #adapt
                mark = amr_instance.vcesmark(mesh, u, lb, bracket=[.2, .8])
                mesh = mesh.refine_marked_elements(mark)
            mtoc = time.perf_counter()

                
        elif method == methodlist[4]: # udo + uniform
            CG1, DG0 = amr_instance.spaces(mesh)
            mtic = time.perf_counter()
            h = max(mesh.cell_sizes.dat.data)
            ratio = HError/(h**2)
            switch = math.isclose(ratio, 1, rel_tol=.1)
            if HError < h**2 or switch:  # uniform
                mark = Function(DG0).interpolate(Constant(1.0))
                mesh = mesh.refine_marked_elements(mark)
                
            else:  # adapt
                mark = amr_instance.udomarkParallel(mesh, u, lb, n=3)
                mesh = mesh.refine_marked_elements(mark)
            mtoc = time.perf_counter()

                
        elif method == methodlist[5]: # metric isotropic + hessian
            mtic = time.perf_counter()
            amr_instance.setmetricparameters(target_complexity=vcesHybridVertexCounts[i])# Corresponds to equal averaging of freeboundary and hessian based metrics.
            mesh = amr_instance.metricrefine(mesh, u, lb, weights=[.5, .5])
            mtoc = time.perf_counter()

            
        elif method == methodlist[6]: # vces + br on inactive set
            mtic = time.perf_counter()
            markFB = amr_instance.vcesmark(mesh, u, lb, bracket=[.2, .8])
            resUFL = Constant(0.0) + div(grad(u))
            markBR = amr_instance.BRinactivemark(mesh, u, lb, resUFL, .5, markFB)
            mark = amr_instance.union(markFB, markBR)
            mesh = mesh.refine_marked_elements(mark)
            mtoc = time.perf_counter()

            
        elif method == methodlist[7]: # udo + br on inactive set
            mtic = time.perf_counter()
            markFB = amr_instance.udomarkParallel(mesh, u, lb, n=3)
            resUFL = Constant(0.0) + div(grad(u))
            markBR = amr_instance.BRinactivemark(mesh, u, lb, resUFL, .5, markFB)
            mark = amr_instance.union(markFB, markBR)
            mesh = mesh.refine_marked_elements(mark)
            mtoc = time.perf_counter()

        
        elif method == methodlist[8]: # uniform
            CG1, DG0 = amr_instance.spaces(mesh)
            mtic = time.perf_counter()
            mark = Function(DG0).interpolate(Constant(1.0))
            mesh = mesh.refine_marked_elements(mark)
            mtoc = time.perf_counter()

            
        else:
            raise ValueError(f"Method not implemented: {args.method}")
        
        
        print(f"Ran {method} refinement on iteration {i}. Solve: {toc - tic} Refinement: {mtoc - mtic}")

        data.append({
            'L2': L2Error,
            'Jaccard': JError,
            'Hausdorff': HError,
            'Elements': ne,
            'Vertices': nv,
            'SizeMin': hmin,
            'SizeMax': hmax,
            'MeshTime': mtoc - mtic,
            'SolveTime' : toc - tic
        })

    # Ensure the 'Results' directory exists
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    # Construct the output file path
    OutputFile = os.path.join(
        results_dir, f"{method}.csv")

    # Create a DataFrame and save it to the specified path
    df = pd.DataFrame(data)
    df.to_csv(OutputFile, index=False)
