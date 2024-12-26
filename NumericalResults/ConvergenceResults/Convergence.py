# Import Firedrake and Netgen
from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from viamr.utility import ObstacleProblem
import math
import argparse
import pandas as pd
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run convergence script with specified parameters.")
    parser.add_argument('-i', '--initTriHeight', type=float,
                        default=0.45, help='Initial triangle height')
    parser.add_argument('-r', '--refinement', type=str,
                        default="Uniform", help='Refinement type')
    parser.add_argument('-m', '--maxIterations', type=int,
                        default=7, help='Maximum number of iterations')
    parser.add_argument('-a', '--amrMethod', type=str,
                        default="udo", help='AMR method to use')

    args = parser.parse_args()

    problem_instance = ObstacleProblem(TriHeight=args.initTriHeight)
    amr_instance = VIAMR()
    meshHistory = [None]
    u = None

    with CheckpointFile("ExactSolution.h5", 'r') as afile:
        # The default name for checkpointing a netgen mesh is not the same as a firedrake mesh
        exactMesh = afile.load_mesh('Default')
        exactU = afile.load_function(exactMesh, "ExactU")
    VTKFile("test.pvd").write(exactU)
    exactV = FunctionSpace(exactMesh, "CG", 1)

    exactPsi, psiufl, _ = problem_instance.sphere_problem(exactMesh, exactV)
    exactElementIndicator = amr_instance.elemactive(exactU, exactPsi)
    _, exactFreeBoundaryEdges = amr_instance.freeboundarygraph(
        exactU, exactPsi)

    data = []
    for i in range(args.maxIterations):
        # solution gets overwritten; never stored
        u, lb, mesh = problem_instance.solveProblem(
            mesh=meshHistory[i], u=u)

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
        _, _, exactU = problem_instance.sphere_problem(
            mesh, u.function_space())
        diffu = Function(u.function_space()).interpolate(u - exactU)
        L2Error = sqrt(assemble(dot(diffu, diffu) * dx))

        # FIXME: investigate this further, this is giving weird results.
        # Compute L2 Error (using conservative projection to finer mesh)
        # proju = Function(exactV).project(u)
        # L2Error = sqrt(
        #    assemble(dot((proju - exactU), (proju - exactU)) * dx(exactMesh)))

        # Refine
        # Hybrid Refinement Strategy:
        # 1. Compute ratio between Hausdorff Error and h^2
        # 2. If Hausdorff Error < h^2 or ratio is within .1 of 1:
        #       apply uniform
        #    Else:
        #        apply AMR
        h = max(mesh.cell_sizes.dat.data)
        CG1, DG0 = amr_instance.spaces(mesh)
        if args.refinement == "Hybrid":
            print(f"Running {args.refinement} scheme:{i}")
            ratio = HError/(h**2)
            switch = math.isclose(ratio, 1, rel_tol=.1)

            if HError < h**2 or switch:
                print("Uniform")
                mark = Function(DG0).interpolate(Constant(1.0))
                nextmesh = mesh.refine_marked_elements(mark)
                mesh = nextmesh
                meshHistory.append(mesh)

            else:
                print("Adaptive")
                if args.amrMethod == "udo":
                    mark = amr_instance.udomark(mesh, u, lb, n=3)
                elif args.amrMethod == "vces":
                    mark = amr_instance.vcesmark(mesh, u, lb, bracket=[.2, .8])

                nextmesh = mesh.refine_marked_elements(mark)
                mesh = nextmesh
                meshHistory.append(mesh)

        elif args.refinement == "Uniform":
            print(f"Running {args.refinement} scheme:{i}")
            mark = Function(DG0).interpolate(Constant(1.0))
            nextmesh = mesh.refine_marked_elements(mark)
            mesh = nextmesh
            meshHistory.append(mesh)

        elif args.refinement == "Adaptive":
            print(f"Running {args.refinement} scheme:{i}")
            if args.amrMethod == "udo":
                mark = amr_instance.udomark(mesh, u, lb, n=3)
            elif args.amrMethod == "vces":
                mark = amr_instance.vcesmark(mesh, u, lb, bracket=[.2, .8])

            nextmesh = mesh.refine_marked_elements(mark)
            mesh = nextmesh
            meshHistory.append(mesh)

        else:
            raise ValueError(f"Unknown refinement type: {args.refinement}")

        data.append({
            'L2': L2Error,
            'Jaccard': JError,
            'Hausdorff': HError,
            'Elements': ne,
            'Vertices': nv,
            'sizes': (hmin, hmax)
        })

    # Ensure the 'Results' directory exists
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    # Construct the output file path
    OutputFile = os.path.join(
        results_dir, f"{args.refinement}_{args.amrMethod}.csv")

    # Create a DataFrame and save it to the specified path
    df = pd.DataFrame(data)
    df.to_csv(OutputFile, index=False)
