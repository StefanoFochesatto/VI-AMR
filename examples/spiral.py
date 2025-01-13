from firedrake import *
from viamr import VIAMR
from viamr.utility import SpiralObstacleProblem
from firedrake.output import VTKFile

u = None
problem = SpiralObstacleProblem(TriHeight=.10)
amr = VIAMR()
mesh = problem.setInitialMesh()
meshHist = [mesh]
for i in range(5):
    mesh = meshHist[i]
    u, lb = problem.solveProblem(mesh=mesh, u=u)
    mark = amr.udomark(mesh, u, lb, n=2)
    mesh = mesh.refine_marked_elements(mark)
    meshHist.append(mesh)

VTKFile('spiral.pvd').write(u)
print('done')
