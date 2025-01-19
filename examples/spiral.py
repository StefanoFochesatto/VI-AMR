from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from viamr.utility import SpiralObstacleProblem

levels = 4
outfile = "result_spiral.pvd"

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
    # alternative:  mark = amr.vcesmark(mesh, u, lb)
    mesh = mesh.refine_marked_elements(mark)
    meshHist.append(mesh)

V = u.function_space()
gap = Function(V, name="gap = u-lb").interpolate(u - lb)

print(f'done: {V.dim()} nodes ... writing to {outfile} ...')
VTKFile(outfile).write(u, lb, gap)
