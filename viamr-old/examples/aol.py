from firedrake import *
from viamr import VIAMR

mesh = RectangleMesh(6, 12, 0.5, 1.0)
x, y = SpatialCoordinate(mesh)
r = (x + 1.0) ** 2 + y ** 2
uexact = conditional(r < 2.0, 0.25 * r - 0.5 - 0.5 * ln(0.5 * r), 0.0)

V = FunctionSpace(mesh, "CG", 1)
uh, vh = Function(V), TestFunction(V)
F = inner(grad(uh), grad(vh)) * dx - Constant(-1) * vh * dx
bcs = DirichletBC(V, Function(V).interpolate(uexact), "on_boundary")
problem = NonlinearVariationalProblem(F, uh, bcs)

sp = {"snes_type": "vinewtonrsls"}
solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
psih = Function(V).interpolate(0.0)
INF = Function(V).interpolate(Constant(PETSc.INFINITY))
solver.solve(bounds=(psih, INF))

amr = VIAMR()
mark = amr.vcdmark(uh, psih)
VTKFile("mesh.pvd").write(uh, mark)

refinedmesh = amr.refinemarkedelements(mesh, mark)
VTKFile("refinedmesh.pvd").write(refinedmesh)
