from firedrake import *
from viamr import VIAMR

mesh0 = RectangleMesh(6, 12, 0.5, 1.0)
x, y = SpatialCoordinate(mesh0)
rr = (x + 1.0) ** 2 + y**2
uex = conditional(rr < 2.0, 0.25 * rr - 0.5 - 0.5 * ln(0.5 * rr), 0.0)

V = FunctionSpace(mesh0, "CG", 1)
u, v = Function(V), TestFunction(V)
F = inner(grad(u), grad(v)) * dx - Constant(-1) * v * dx
bcs = DirichletBC(V, Function(V).interpolate(uex), "on_boundary")
problem = NonlinearVariationalProblem(F, u, bcs)

p = {"snes_type": "vinewtonrsls"}
solver = NonlinearVariationalSolver(problem, solver_parameters=p)
lb = Function(V).interpolate(Constant(0.0))
ub = Function(V).interpolate(Constant(PETSc.INFINITY))
solver.solve(bounds=(lb, ub))

amr = VIAMR()
mark = amr.vcdmark(mesh0, u, lb)
VTKFile("mesh0.pvd").write(u, mark)

mesh1 = amr.refinemarkedelements(mesh0, mark)
VTKFile("mesh1.pvd").write(mesh1)
