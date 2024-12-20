# Import Firedrake and Netgen
from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from TestProblem import ObstacleProblem
import os

if __name__ == "__main__":
    problem_instance = ObstacleProblem(TriHeight=.5)
    amr_instance = VIAMR()
    solutionHistory = [None]
    meshHistory = [None]

    # Temporary for debugging purposes
    # os.chdir("/home/stefano/Desktop/VI-AMR/NumericalResults/ConvergenceResults")
    with CheckpointFile("ExactSolution.h5", 'r') as afile:
        # The default name for checkpointing a netgen mesh is not the same as a firedrake mesh
        exactMesh = afile.load_mesh('Default')
        exactU = afile.load_function(exactMesh, "ExactU")

    exactV = FunctionSpace(exactMesh)
    exactPsi, psiufl, _ = problem_instance.sphere_problem(exactV, exactMesh)
    exactElementIndicator = amr_instance.elemactive(exactU, exactPsi)
    _, exactFreeBoundaryEdges = amr_instance.freeboundarygraph(
        exactU, exactPsi)

    for i in range(max_iterations):
        u, lb, mesh = problem_instance.solveProblem(
            mesh=meshHistory[i], u=solutionHistory[i])

        solElementIndicator = amr_instance.elemactive(u, lb)
        # Compute Jaccard index
        JError = amr_instance.jaccard(
            solElementIndicator, exactElementIndicator)

        _, solFreeBoundaryEdges = amr_instance.freeboundarygraph(u, lb)
        # Compute Hausdorff error
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

        if Refinement == "Hybrid":
            ratio = HError/(h**2)
            switch = math.isclose(ratio, 1, rel_tol=.1)

            if HError < h**2 or switch:
                mark = MarkUDO(mesh, u, lb, i, Refinement, 0)
                nextmesh = mesh.refine_marked_elements(mark)
                meshHierarchy.append(nextmesh)
                h = h*(1/2)
                count.append(0)

            else:
                mark = MarkUDO(mesh, u, lb, i, Refinement, n)
                nextmesh = mesh.refine_marked_elements(mark)
                meshHierarchy.append(nextmesh)
                count.append(1)
        else:
            mark = MarkUDO(mesh, u, lb, i, Refinement, n)
            nextmesh = mesh.refine_marked_elements(mark)
            meshHierarchy.append(nextmesh)
