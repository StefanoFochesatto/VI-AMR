# Testing Script for Parallel UDO Ideas
from firedrake import *
from firedrake.output import VTKFile

from viamr import VIAMR
from viamr import SpiralObstacleProblem

initTriHeight = .05
problem_instance = SpiralObstacleProblem(TriHeight=initTriHeight)
mesh = problem_instance.setInitialMesh()
u, lb = problem_instance.solveProblem(mesh=mesh, u=None)
z = VIAMR()
mark = z.udomark(mesh, u, lb, n=3)
mark.rename('mark')

with CheckpointFile("SerialUDO.h5", 'w') as afile:
    afile.save_mesh(mesh)
    afile.save_function(mark)
