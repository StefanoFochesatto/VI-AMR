from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
print = PETSc.Sys.Print # enables correct printing in parallel
from viamr import VIAMR
from viamr.utility import SphereObstacleProblem

levels = 5
h_initial = 0.10
outfile = "result_sphere.pvd"

problem = SphereObstacleProblem(TriHeight=h_initial)
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
    # alternative:  mark = amr.udomark(mesh, u, lb, n=2)
    mark = amr.vcdmark(mesh, u, lb, bracket=[0.2, 0.9])
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
