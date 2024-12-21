# Import Firedrake and Netgen
from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from TestProblem import ObstacleProblem
import os
import math


if __name__ == "__main__":
    problem_instance = ObstacleProblem(TriHeight=.5)
    amr_instance = VIAMR()
    meshHistory = [None]
    u = None
    Refinement = "Hybrid"
    max_iterations = 7
    # Temporary for debugging purposes
    os.chdir("/home/stefano/Desktop/VI-AMR/NumericalResults/ConvergenceResults")
    with CheckpointFile("ExactSolution.h5", 'r') as afile:
        # The default name for checkpointing a netgen mesh is not the same as a firedrake mesh
        exactMesh = afile.load_mesh('Default')
        exactU = afile.load_function(exactMesh, "ExactU")

    exactV = FunctionSpace(exactMesh, "CG", 1)
    exactPsi, psiufl, _ = problem_instance.sphere_problem(exactMesh, exactV)
    exactElementIndicator = amr_instance.elemactive(exactU, exactPsi)
    _, exactFreeBoundaryEdges = amr_instance.freeboundarygraph(
        exactU, exactPsi)

    for i in range(max_iterations):
        # solution gets overwritten; never stored
        u, lb, mesh = problem_instance.solveProblem(
            mesh=meshHistory[i], u=u)

        # Compute Jaccard index
        solElementIndicator = amr_instance.elemactive(u, lb)
        JError = amr_instance.jaccard(
            solElementIndicator, exactElementIndicator)

        # Compute Hausdorff error
        _, solFreeBoundaryEdges = amr_instance.freeboundarygraph(u, lb)
        HError = amr_instance.hausdorff(
            solFreeBoundaryEdges, exactFreeBoundaryEdges)

        # Compute L2 Error (using conservative projection to finer mesh)
        proju = Function(exactV).project(u)
        L2Error = sqrt(
            assemble(dot((proju - exactU), (proju - exactU)) * dx(exactMesh)))

        # Refine
        # Hybrid Refinement Strategy:
        # 1. Compute ratio between Hausdorff Error and h^2
        # 2. If Hausdorff Error < h^2 or ratio is within .1 of 1:
        #       apply uniform
        #    Else:
        #        apply AMR
        h = max(mesh.cell_sizes.dat.data)
        CG1, DG0 = amr_instance.spaces(mesh)
        if Refinement == "Hybrid":
            print(f"Running {Refinement} scheme:{i}")
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
                mark = amr_instance.udomark(mesh, u, lb, n=3)
                nextmesh = mesh.refine_marked_elements(mark)
                mesh = nextmesh
                meshHistory.append(mesh)

        elif Refinement == "Uniform":
            print(f"Running {Refinement} scheme:{i}")
            mark = Function(DG0).interpolate(Constant(1.0))
            nextmesh = mesh.refine_marked_elements(mark)
            mesh = nextmesh
            meshHistory.append(mesh)

        elif Refinement == "Adaptive":
            print(f"Running {Refinement} scheme:{i}")
            mark = amr_instance.udomark(mesh, u, lb, n=3)
            nextmesh = mesh.refine_marked_elements(mark)
            mesh = nextmesh
            meshHistory.append(mesh)

        else:
            raise ValueError(f"Unknown refinement type: {Refinement}")
