from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
from viamr import VIAMR


amr = VIAMR()
mesh = BoxMesh(5, 5, 10, 1, 1, 1)
DG0 = FunctionSpace(mesh, "DG", 0)
(x, y, z) = SpatialCoordinate(mesh)
indicator = Function(DG0).interpolate(conditional(x>.5, 1, 0))

refinedmesh = amr.refinemarkedelements(mesh, indicator)

VTKFile("3d.pvd").write(refinedmesh)

