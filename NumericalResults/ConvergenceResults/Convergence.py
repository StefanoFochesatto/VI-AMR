# Import Firedrake and Netgen
from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from TestProblem import ObstacleProblem

if __name__ == "__main__":
    problem_instance = ObstacleProblem(TriHeight=.5)
    solutionHistory = [None]
    meshHistory = [None]

    with CheckpointFile("ExactSolution.h5", 'r') as afile:
        ExactMesh = afile.load_mesh("ExactMesh")
        ExactU = afile.load_function(ExactMesh, "u")

    for i in range(max_iterations):
        problem_instance.solveProblem(
            mesh=meshHistory[i], u=solutionHistory[i])
        amr_instance = VIAMR()

        # Compute Jaccard index

        # Compute Hausdorff error

        # Compute L2 Error

        # Refine
        # Hybrid Refinement Strategy:
        # 1. Compute ratio between Hausdorff Error and h^2
        # 2. If Hausdorff Error < h^2 or ratio is within .1 of 1:
        #       apply uniform
        #    Else:
        #        apply AMR

        if Refinement == "Hybrid":
            ratio = HausdorffError/(h**2)
            switch = math.isclose(ratio, 1, rel_tol=.1)

            if HausdorffError < h**2 or switch:
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
