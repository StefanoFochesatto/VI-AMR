from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from utility import LShapedDomainProblem
import os


problem = LShapedDomainProblem()
amr = VIAMR()
ExactMesh = problem.setInitialMesh("lshapedSolution.msh")
ExactU = None

nv, ne, hmin, hmax = amr.meshsizes(ExactMesh)
for i in range(5):
    ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU)
    ExactU.rename("ExactU")
    DG0 = FunctionSpace(ExactMesh, "DG", 0)
    resUFL = -div(grad(ExactU))
    FBmark = amr.udomark(ExactU, lb, n  = 1)
    (BRmark, _, _) = amr.brinactivemark(ExactU, lb, resUFL,method = "max", theta=0.7)
    mark = amr.unionmarks(FBmark, BRmark)
    ExactMesh = amr.refinemarkedelements(ExactMesh, mark)
    
    
for i in range(4):
    ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU)
    ExactU.rename("ExactU")
    DG0 = FunctionSpace(ExactMesh, "DG", 0)
    resUFL = -div(grad(ExactU))
    (BRmark, _, _) = amr.brinactivemark(ExactU, lb, resUFL,method = "max", theta=0.7)
    ExactMesh = amr.refinemarkedelements(ExactMesh, BRmark)

amr.setmetricparameters(target_complexity=15000, h_min=1.0e-8, h_max=10.0)
ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU)
ExactMesh = amr.adaptaveragedmetric(ExactMesh, ExactU, lb, gamma = .95)
ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU)





active = amr.elemactive(ExactU, lb)
mesh = ExactU.function_space().mesh()
# Get the spatial coordinates from the mesh
x = ufl.SpatialCoordinate(ExactU.function_space().mesh())

rect_condition = ufl.And(ufl.And(x[0] >= -2.0, x[0] <= 2.7),
                         ufl.And(x[1] >= -2.75, x[1] <= 1.5))


zoom = Function(FunctionSpace(mesh, "DG", 0)).interpolate(ufl.conditional(rect_condition, 1.0, 0.0))

submesh = amr._filtermesh(mesh, zoom)


activezoom = Function(FunctionSpace(submesh, "DG", 0)).interpolate(active)
VTKFile("lshapedzoomed.pvd").write(activezoom)

VTKFile("lshaped.pvd").write(ExactU, active)
