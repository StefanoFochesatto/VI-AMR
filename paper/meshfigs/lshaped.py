from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from paper.convergence.utility import LShapedDomainProblem
import os


problem = LShapedDomainProblem()
amr = VIAMR()
ExactMesh = problem.setInitialMesh("lshapedSolution.msh")
ExactU = None

nv, ne, hmin, hmax = amr.meshsizes(ExactMesh)
for i in range(7):
    ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU)
    ExactU.rename("ExactU")
    if i == 100:
        # Run one metric refine to have substantially different mesh from convergence runs
        amr.setmetricparameters(target_complexity=nv/6)
        ExactMesh = amr.adaptaveragedmetric(
            ExactMesh, ExactU, lb
        )
    else:
        DG0 = FunctionSpace(ExactMesh, "DG", 0)
        resUFL = -div(grad(ExactU))
        FBmark = amr.udomark(ExactU, lb, n  = 1)
        (BRmark, _, _) = amr.brinactivemark(ExactU, lb, resUFL,method = "total", theta=0.1)
        mark = amr.unionmarks(FBmark, BRmark)
        ExactMesh = amr.refinemarkedelements(ExactMesh, mark)

active = amr._eleminactive(ExactU, lb)

mesh = ExactU.function_space().mesh()
# Get the spatial coordinates from the mesh
x = ufl.SpatialCoordinate(ExactU.function_space().mesh())

rect_condition = ufl.And(ufl.And(x[0] >= -2.0, x[0] <= 2.7),
                         ufl.And(x[1] >= -2.3, x[1] <= 1.5))


zoom = Function(FunctionSpace(mesh, "DG", 0)).interpolate(ufl.conditional(rect_condition, 1.0, 0.0))

submesh = amr._filtermesh(mesh, zoom)


activezoom = Function(FunctionSpace(submesh, "DG", 0)).interpolate(active)
VTKFile("lshapedzoomed.pvd").write(activezoom)

VTKFile("lshaped.pvd").write(active)
