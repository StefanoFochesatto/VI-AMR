from firedrake import *
from viamr import VIAMR
from viamr.utility import SpiralObstacleProblem
from firedrake.output import VTKFile

levels = 5

problem = SpiralObstacleProblem(TriHeight=.10)
amr = VIAMR()
mesh = problem.setInitialMesh()
meshHist = [mesh]
u = None
for i in range(levels + 1):
    mesh = meshHist[i]
    u, lb = problem.solveProblem(mesh=mesh, u=u)
    if i == levels:
        break
    mark = amr.udomark(mesh, u, lb, n=2)
    mesh = mesh.refine_marked_elements(mark)
    meshHist.append(mesh)

V = u.function_space()
gap = Function(V, name="gap = u-lb").interpolate(u - lb)

print('done ... writing to result.pvd ...')
VTKFile('result.pvd').write(u, lb, gap)
