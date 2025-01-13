# Testing Script for Parallel UDO Ideas
from firedrake import *
from firedrake.output import VTKFile

from viamr import VIAMR
from viamr import SphereObstacleProblem

initTriHeight = .05
problem_instance = SphereObstacleProblem(TriHeight=initTriHeight)
u, lb, mesh = problem_instance.solveProblem(mesh=None, u=None)
z = VIAMR()
mark = z.udomark(mesh, u, lb, n=3)
mark.rename('mark')

with CheckpointFile("SerialUDO.h5", 'w') as afile:
    afile.save_mesh(mesh)
    afile.save_function(mark)
