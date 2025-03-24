from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from viamr.utility import LShapedDomainProblem
import os

# For debugging purposes
#os.chdir("/home/stefano/Desktop/VI-AMR/NumericalResults/ConvergenceResults")
problem = LShapedDomainProblem(TriHeight=.25)
amr = VIAMR()
ExactMesh = problem.setInitialMesh("lshapedSolution.msh")
ExactU = None

nv, ne, hmin, hmax = amr.meshsizes(ExactMesh)
for i in range(5):
    ExactU, lb = problem.solveProblem(mesh = ExactMesh, u = ExactU)
    ExactU.rename("ExactU")
    if i == 0:
        # Run one metric refine to have substantially different mesh from convergence runs
        amr.setmetricparameters(
            target_complexity=nv)
        ExactMesh = amr.metricrefine(ExactMesh, ExactU, lb, weights=[.5, .5])
        
    else:
        DG0 = FunctionSpace(ExactMesh, "DG", 0)
        resUFL = Constant(0.0) + div(grad(ExactU))
        FBmark = amr.vcesmark(ExactMesh, ExactU, lb)
        BRmark = amr.BRinactivemark(ExactMesh, ExactU, lb, resUFL=resUFL, theta = .50, markFB=FBmark)
        mark = amr.union(FBmark, BRmark)
        ExactMesh = amr.refinemarkedelements(ExactMesh, mark)
        



mh = MeshHierarchy(ExactMesh, 4)
ExactMesh = mh[-1]

ExactU = Function(FunctionSpace(ExactMesh, "CG", 1)).interpolate(ExactU)

ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU, FASCD=True)
ExactU.rename("ExactU")

VTKFile("Test.pvd").write(ExactU)
# Open issue won't run in parallel with netgen mesh: https://github.com/firedrakeproject/firedrake/issues/3783
with CheckpointFile("ExactSolutionLShaped.h5", 'w') as afile:
    afile.save_mesh(ExactMesh)
    afile.save_function(ExactU)
