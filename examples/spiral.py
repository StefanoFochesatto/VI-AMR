from firedrake import *
from viamr import VIAMR
from viamr.utility import SpiralObstacleProblem
from firedrake.output import VTKFile

problem = SpiralObstacleProblem(TriHeight=.10)
amr = VIAMR()
mesh = problem.setInitialMesh()
meshHist = [mesh]
u = None
for i in range(5):
    mesh = meshHist[i]
    u, lb = problem.solveProblem(mesh=mesh, u=u)
    mark = amr.udomark(mesh, u, lb, n=2)
    mesh = mesh.refine_marked_elements(mark)
    meshHist.append(mesh)

gap = Function(u.function_space(), name="gap = u-lb").interpolate(u - lb)
print('done ... writing to result.pvd ...')
VTKFile('result.pvd').write(u, gap)
