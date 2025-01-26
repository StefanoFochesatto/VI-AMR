from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from viamr.utility import SpiralObstacleProblem

problem = SpiralObstacleProblem()
hierarchy = [problem.setInitialMesh()]
amr = VIAMR()
u = None
for i in range(4):
    mesh = hierarchy[i]
    u, psi = problem.solveProblem(mesh=mesh, u=u)
    if i == 3:
        print(f'final mesh: N={mesh.num_cells()} elements')
        gap = Function(u.function_space()).interpolate(u - psi)
        VTKFile("result.pvd").write(u, psi, gap)
        break
    mark = amr.udomark(mesh, u, psi, n=2)
    mesh = mesh.refine_marked_elements(mark)
    hierarchy.append(mesh)
