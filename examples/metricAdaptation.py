from netgen.geom2d import SplineGeometry
import numpy as np
from firedrake import *
from animate import *   # see README.md regarding this dependency
from firedrake.output import VTKFile
from viamr.utility import SphereObstacleProblem
from viamr import VIAMR


problem = SphereObstacleProblem(TriHeight=0.1)
mesh = problem.setInitialMesh()
u, lb = problem.solveProblem(mesh=mesh, u=None)

viamr = VIAMR()
adaptedMesh = viamr.metricrefine(mesh, u, lb)
u, lb = problem.solveProblem(mesh=adaptedMesh, u=u)
gap = Function(u.function_space(), name="gap = u - lb")
gap.interpolate(u - lb)

# FIXME evaluate exact solution

VTKFile("results.pvd").write(u, lb, gap)
