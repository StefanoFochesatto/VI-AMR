from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from paper.convergence.utility import SpiralObstacleProblem
import os


problem = SpiralObstacleProblem(TriHeight=0.05)
mesh = problem.setInitialMesh()
u = None
amr_instance = VIAMR()

for i in range(5):
    u, lb = problem.solveProblem(mesh=mesh, u=u)
    markFB = amr_instance.udomark(u, lb, n = (i + 2))
    resUFL = -div(grad(u))
    (markBR, _, _) = amr_instance.brinactivemark(u, lb, resUFL, theta=0.8)
    mark = amr_instance.unionmarks(markFB, markBR)
    mesh = amr_instance.refinemarkedelements(mesh, mark)
    
active = amr_instance._eleminactive(u, lb)
   
VTKFile("spiral.pvd").write(active)
    