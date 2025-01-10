from firedrake import *
from viamr import VIAMR
from viamr.utility import SphereObstacleProblem

u = None
problem = SphereObstacleProblem(TriHeight=.05)
amr = VIAMR()
mesh = problem.setInitialMesh()
meshHist = [mesh]
for i in range(5):
    mesh = meshHist[i]
    u, lb = problem.solveProblem(mesh=mesh, u=u)
    mark = amr.udomark(mesh, u, lb, n=2)
    mesh = mesh.refine_marked_elements(mark)
    meshHist.append(mesh)
print('done')
