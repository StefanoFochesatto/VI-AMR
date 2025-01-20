from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from viamr.utility import SphereObstacleProblem

levels = 4
outfile = "result_sphere.pvd"

problem = SphereObstacleProblem(TriHeight=.10)
amr = VIAMR()
mesh = problem.setInitialMesh()
meshHist = [mesh]
u = None
for i in range(levels + 1):
    mesh = meshHist[i]
    print(f'solving on mesh {i} ...')
    amr.meshreport(mesh)
    spmore = {
        "snes_converged_reason": None,
        "snes_vi_monitor": None,
    }
    u, lb = problem.solveProblem(mesh=mesh, u=u, moreparams=spmore)
    if i == levels:
        break
    mark = amr.udomark(mesh, u, lb, n=2)
    # alternative:  mark = amr.vcesmark(mesh, u, lb)
    mesh = mesh.refine_marked_elements(mark)
    meshHist.append(mesh)

V = u.function_space()
gap = Function(V, name="gap = u-lb").interpolate(u - lb)
uexact = problem.getExactSolution(V)
uexact.rename("u_exact")
error = Function(V, name="error = |u - u_exact|")
error.interpolate(abs(u - uexact))

print(f'|u - u_exact|_2 = {errornorm(u, uexact):.3e}')
print(f'done ... writing to {outfile} ...')
VTKFile(outfile).write(u, lb, gap, uexact, error)
