from netgen.geom2d import SplineGeometry
import numpy as np
from firedrake import *
from animate import *   # see README.md regarding this dependency
from firedrake.output import VTKFile
from viamr.utility import ObstacleProblem
from viamr import VIAMR


initTriHeight = .1

problem_instance = ObstacleProblem(TriHeight=initTriHeight)
u, lb, mesh = problem_instance.solveProblem(mesh=None, u=None)
viamr = VIAMR()
adaptedMesh = viamr.metricrefine(mesh, u, lb)


VTKFile("results.pvd").write(adaptedMesh)
