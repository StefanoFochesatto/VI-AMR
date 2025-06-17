# Import necessary modules from Firedrake
from firedrake import *
from firedrake.output import VTKFile
from paper.convergence.utility import SphereObstacleProblem
from viamr import VIAMR

# for debugging
#import os
#os.chdir("/home/stefano/Desktop/VI-AMR/NumericalResults/ConvergenceResults")


problem = SphereObstacleProblem(TriHeight=.25)
amr = VIAMR()
ExactMesh = problem.setInitialMesh()
ExactU = None

nv, ne, hmin, hmax = amr.meshsizes(ExactMesh)
for i in range(4):
    ExactU, lb = problem.solveProblem(mesh = ExactMesh, u = ExactU)
    ExactU.rename("ExactU")

        
    DG0 = FunctionSpace(ExactMesh, "DG", 0)

    resUFL = -div(grad(ExactU))
    FBmark = amr.vcdmark(ExactU, lb)
    (BRmark, _, _) = amr.brinactivemark(ExactU, lb, resUFL, theta=.5)
    mark = amr.unionmarks(FBmark, BRmark)
    ExactMesh = amr.refinemarkedelements(ExactMesh, mark)
    



mh = MeshHierarchy(ExactMesh, 4)
ExactMesh = mh[-1]

ExactU = Function(FunctionSpace(ExactMesh, "CG", 1)).interpolate(ExactU)

ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU, FASCD=True)
ExactU.rename("ExactU")

# Open issue won't run in parallel with netgen mesh: https://github.com/firedrakeproject/firedrake/issues/3783
with CheckpointFile("ExactSolution.h5", 'w') as afile:
    afile.save_mesh(ExactMesh)
    afile.save_function(ExactU)
