# Import Firedrake and Netgen
from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from paper.convergence.utility import LShapedDomainProblem
import time
import math
import argparse
import pandas as pd
import os
from animate import adapt


vcdHybridVertexCounts = [229, 425, 860, 1616, 3219, 6385, 116]
vcdAdaptVertexCounts = [178, 328, 618, 1207, 2465, 4855, 116]


methodlist = [
    "vcd",
    "udo",
    "metricIso",
    "vcdUnif",
    "udoUnif",
    "metricIsoHess",
    "vcdBR",
    "udoBR",
    "uniform",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run convergence script with specified parameters."
    )
    parser.add_argument(
        "-t",
        "--initTriHeight",
        type=float,
        default=0.45,
        help="Initial triangle height",
    )
    parser.add_argument(
        "-i",
        "--maxIterations",
        type=int,
        default=7,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "-m", "--method", type=str, default="uniform", help="string indicating method"
    )

    args = parser.parse_args()
    method = args.method

    problem_instance = LShapedDomainProblem(TriHeight=args.initTriHeight)
    amr_instance = VIAMR()
    mesh = problem_instance.setInitialMesh("lshaped.msh")
    u = None

    # for debugging purposes
    # os.chdir("/home/stefano/Desktop/VI-AMR/NumericalResults/ConvergenceResults")

    # Load in data about exact solution
    with CheckpointFile("ExactSolutionLShaped.h5", "r") as afile:
        # The default name for checkpointing a netgen mesh is not the same as a firedrake mesh # this might be fixed in new firedrake builds
        exactMesh = afile.load_mesh()
        exactU = afile.load_function(exactMesh, "ExactU")
    exactV = FunctionSpace(exactMesh, "CG", 1)

    exactPsi = problem_instance.getObstacle(exactV)
    exactElementIndicator = amr_instance._elemactive(exactU, exactPsi)
    _, exactFreeBoundaryEdges = amr_instance.freeboundarygraph(exactU, exactPsi)

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
        solElementIndicator = amr_instance._elemactive(u, lb)
        JError = amr_instance.jaccard(solElementIndicator, exactElementIndicator)

        # Compute Hausdorff error
        _, solFreeBoundaryEdges = amr_instance.freeboundarygraph(u, lb)
        HError = amr_instance.hausdorff(solFreeBoundaryEdges, exactFreeBoundaryEdges)

        # Compute L2 error
        # L2Error = sqrt(assemble(dot(diffu, diffu) * dx))
        uProj = Function(exactU.function_space()).project(u)
        L2Error = errornorm(exactU, uProj, norm_type="L2")
        # Compute H1 error
        H1Error = errornorm(exactU, uProj, norm_type="H1")

        # FIXME: investigate this further, this is giving weird results.
        # Compute L2 Error (using conservative projection to finer mesh)
        # proju = Function(exactV).project(u)
        # L2Error = sqrt(
        #    assemble(dot((proju - exactU), (proju - exactU)) * dx(exactMesh)))

        # Big switch style statement for methods
        if method == methodlist[0]:  # vcd
            mtic = time.perf_counter()
            mark = amr_instance.vcdmark(u, lb, bracket=[0.2, 0.8])
            mtoc = time.perf_counter()
            rtic = time.perf_counter()
            mesh = amr_instance.refinemarkedelements(mesh, mark)
            rtoc = time.perf_counter()

        elif method == methodlist[1]:  # udo
            mtic = time.perf_counter()
            mark = amr_instance.udomark(u, lb, n=3)
            mtoc = time.perf_counter()
            rtic = time.perf_counter()
            mesh = amr_instance.refinemarkedelements(mesh, mark)
            rtoc = time.perf_counter()

        elif method == methodlist[2]:  # metric isotropic only
            mtic = time.perf_counter()
            amr_instance.setmetricparameters(
                target_complexity=vcdAdaptVertexCounts[i]
            )  # Corresponds to only freeboundary metric applied
            VImetric = amr_instance.adaptaveragedmetric(
                mesh, u, lb, gamma = 1, metric=True
            )
            mtoc = time.perf_counter()
            rtic = time.perf_counter()
            mesh = adapt(mesh, VImetric)
            rtoc = time.perf_counter()

        elif method == methodlist[3]:  # vcd + uniform
            CG1, DG0 = amr_instance.spaces(mesh)
            mtic = time.perf_counter()
            h = max(mesh.cell_sizes.dat.data)
            ratio = HError / (h**2)
            switch = math.isclose(ratio, 1, rel_tol=0.1)
            if HError < h**2 or switch:  # uniform
                mark = Function(DG0).interpolate(Constant(1.0))
                mtoc = time.perf_counter()
                rtic = time.perf_counter()
                mesh = amr_instance.refinemarkedelements(mesh, mark, isUniform=True)
                rtoc = time.perf_counter()

            else:  # adapt
                mark = amr_instance.vcdmark(u, lb, bracket=[0.2, 0.8])
                mtoc = time.perf_counter()
                rtic = time.perf_counter()
                mesh = amr_instance.refinemarkedelements(mesh, mark)
                rtoc = time.perf_counter()

        elif method == methodlist[4]:  # udo + uniform
            CG1, DG0 = amr_instance.spaces(mesh)
            mtic = time.perf_counter()
            h = max(mesh.cell_sizes.dat.data)
            ratio = HError / (h**2)
            switch = math.isclose(ratio, 1, rel_tol=0.1)
            if HError < h**2 or switch:  # uniform
                mtoc = time.perf_counter()
                mark = Function(DG0).interpolate(Constant(1.0))
                rtic = time.perf_counter()
                mesh = amr_instance.refinemarkedelements(mesh, mark, isUniform=True)
                rtoc = time.perf_counter()

            else:  # adapt
                mark = amr_instance.udomark(u, lb, n=3)
                mtoc = time.perf_counter()
                rtic = time.perf_counter()
                mesh = amr_instance.refinemarkedelements(mesh, mark)
                rtoc = time.perf_counter()

        elif method == methodlist[5]:  # metric isotropic + hessian
            mtic = time.perf_counter()
            amr_instance.setmetricparameters(
                target_complexity=vcdHybridVertexCounts[i]
            )  # Corresponds to equal averaging of freeboundary and hessian based metrics.
            VImetric = amr_instance.adaptaveragedmetric(
                mesh, u, lb, gamma = .5, hessian=True, metric=True
            )
            mtoc = time.perf_counter()
            rtic = time.perf_counter()
            mesh = adapt(mesh, VImetric)
            rtoc = time.perf_counter()

        elif method == methodlist[6]:  # vcd + br on inactive set
            mtic = time.perf_counter()
            markFB = amr_instance.vcdmark(u, lb, bracket=[0.2, 0.8])
            resUFL = -div(grad(u))
            (markBR, _, _) = amr_instance.brinactivemark(u, lb, resUFL, theta=0.5)
            mark = amr_instance.unionmarks(markFB, markBR)
            mtoc = time.perf_counter()
            rtic = time.perf_counter()
            mesh = amr_instance.refinemarkedelements(mesh, mark)
            rtoc = time.perf_counter()

        elif method == methodlist[7]:  # udo + br on inactive set
            mtic = time.perf_counter()
            markFB = amr_instance.udomark(u, lb, n=3)
            resUFL = -div(grad(u))
            (markBR, _, _) = amr_instance.brinactivemark(u, lb, resUFL, theta=0.5)
            mark = amr_instance.unionmarks(markFB, markBR)
            mtoc = time.perf_counter()
            rtic = time.perf_counter()
            mesh = amr_instance.refinemarkedelements(mesh, mark)
            rtoc = time.perf_counter()

        elif method == methodlist[8]:  # uniform
            CG1, DG0 = amr_instance.spaces(mesh)
            mark = Function(DG0).interpolate(Constant(1.0))
            mtic = time.perf_counter()
            mtoc = time.perf_counter()
            rtic = time.perf_counter()
            mesh = amr_instance.refinemarkedelements(mesh, mark, isUniform=True)
            rtoc = time.perf_counter()

        else:
            raise ValueError(f"Method not implemented: {args.method}")

        print(
            f"Ran {method} refinement on iteration {i}. Solve: {toc - tic} Refinement: {mtoc - mtic}"
        )

        data.append(
            {
                "L2": L2Error,
                "H1": H1Error,
                "Jaccard": JError,
                "Hausdorff": HError,
                "Elements": ne,
                "Vertices": nv,
                "SizeMin": hmin,
                "SizeMax": hmax,
                "PreMeshCompTime": mtoc - mtic,
                "RefineTime": rtoc - rtic,
                "SolveTime": toc - tic,
            }
        )

    # Ensure the 'Results' directory exists
    results_dir = "ResultsLShaped"
    os.makedirs(results_dir, exist_ok=True)

    # Construct the output file path
    OutputFile = os.path.join(results_dir, f"{method}.csv")

    # Create a DataFrame and save it to the specified path
    df = pd.DataFrame(data)
    df.to_csv(OutputFile, index=False)
