from netgen.geom2d import SplineGeometry
import numpy as np
from firedrake import *
from animate import *   # see README.md regarding this dependency
from firedrake.output import VTKFile
from TestProblem import ObstacleProblem
from viamr import VIAMR


class VIAMRMetric(VIAMR):
    def vcesmark(self, *args, **kwargs):
        # Call the original method
        mark = super().vcesmark(*args, **kwargs)
        return mark, u


initTriHeight = .1

problem_instance = ObstacleProblem(TriHeight=initTriHeight)
u, lb, mesh = problem_instance.solveProblem(mesh=None, u=None)
viamr = VIAMRMetric()
_, s = viamr.vcesmark(mesh, u, lb)


V = FunctionSpace(mesh, "CG", 1)
gs = Function(V).project(dot(grad(s), grad(s)))
dim = mesh.topological_dimension()


adapted = viamr.metricrefine(mesh, u, lb)

P1_ten = TensorFunctionSpace(mesh, "CG", 1)
metric = RiemannianMetric(P1_ten)

mp = {
    "dm_plex_metric": {
        "target_complexity": 300.0,
        "p": 1.0,  # normalisation order
        "h_min": 1e-07,  # minimum allowed edge length
        "h_max": 1.0,  # maximum allowed edge length
    }}

metric.set_parameters(mp)
metric.interpolate(abs(gs) * ufl.Identity(dim))
metric.normalise()


adaptedMesh = adapt(mesh,  metric)

VTKFile("result.pvd").write(adapted)
