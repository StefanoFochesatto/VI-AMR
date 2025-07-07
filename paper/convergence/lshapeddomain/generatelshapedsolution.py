from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from paper.convergence.utility import LShapedDomainProblem
import os

# For debugging purposes
#os.chdir("/home/stefano/Desktop/VI-AMR/paper/convergence/lshapeddomain")
problem = LShapedDomainProblem()
amr = VIAMR()
ExactMesh = problem.setInitialMesh("lshapedSolution.msh")
ExactU = None

nv, ne, hmin, hmax = amr.meshsizes(ExactMesh)
for i in range(5):
    ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU)
    ExactU.rename("ExactU")
    if i == 0:
        # Run one metric refine to have substantially different mesh from convergence runs
        amr.setmetricparameters(target_complexity=nv)
        ExactMesh = amr.adaptaveragedmetric(
            ExactMesh, ExactU, lb
        )
    else:
        DG0 = FunctionSpace(ExactMesh, "DG", 0)
        resUFL = -div(grad(ExactU))
        FBmark = amr.vcdmark(ExactU, lb)
        (BRmark, _, _) = amr.brinactivemark(ExactU, lb, resUFL, theta=0.5)
        mark = amr.unionmarks(FBmark, BRmark)
        ExactMesh = amr.refinemarkedelements(ExactMesh, mark)


mh = MeshHierarchy(ExactMesh, 3)
ExactMesh = mh[-1]

ExactU = Function(FunctionSpace(ExactMesh, "CG", 1)).interpolate(ExactU)

ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU, FASCD=True)
ExactU.rename("ExactU")

# Open issue won't run in parallel with netgen mesh: https://github.com/firedrakeproject/firedrake/issues/3783
with CheckpointFile("ExactSolutionLShaped.h5", "w") as afile:
    afile.save_mesh(ExactMesh)
    afile.save_function(ExactU)
