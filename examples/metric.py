from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from viamr.utility import SphereObstacleProblem
from animate import *   # see README.md regarding this dependency

outfile = "result_metric.pvd"

problem = SphereObstacleProblem(TriHeight=0.1)
mesh0 = problem.setInitialMesh()
u0, lb0 = problem.solveProblem(mesh=mesh0, u=None)

amr = VIAMR()
mesh = amr.metricrefine(mesh0, u0, lb0)
u, lb = problem.solveProblem(mesh=mesh, u=u0)

V = u.function_space()
gap = Function(V, name="gap = u-lb").interpolate(u - lb)
uexact = problem.getExactSolution(V)
uexact.rename("u_exact")
error = Function(V, name="error = |u - u_exact|")
error.interpolate(abs(u - uexact))

amr.meshreport(mesh)
print(f'|u - u_exact|_2 = {errornorm(u, uexact):.3e}')
print(f'done ... writing to {outfile} ...')
VTKFile(outfile).write(u, lb, gap, uexact, error)
