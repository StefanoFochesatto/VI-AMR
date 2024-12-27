# modified from https://www.firedrakeproject.org/demos/poisson_mixed.py.html

obstacle = True   # solves Poisson equation if False

from firedrake import *

mesh = UnitSquareMesh(32, 32)

BDM = FunctionSpace(mesh, "BDM", 1)
DG = FunctionSpace(mesh, "DG", 0)
Z = BDM * DG

sigmau = Function(Z)
sigma, u = split(sigmau)
tau, v = TestFunctions(Z)

x, y = SpatialCoordinate(mesh)
frhs = Function(DG).interpolate(10*exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

F = (dot(sigma, tau) + div(tau) * u + div(sigma) * v)*dx - frhs * v * dx

bc0 = DirichletBC(Z.sub(0), as_vector([0.0, -sin(5*x)]), 3)
bc1 = DirichletBC(Z.sub(0), as_vector([0.0, sin(5*x)]), 4)

sigmau.subfunctions[0].interpolate(as_vector([Constant(0.0),Constant(0.0)]))
sigmau.subfunctions[1].interpolate(Constant(1.0))

problem = NonlinearVariationalProblem(F, sigmau, bcs=[bc0, bc1])

# common solver parameters
sp = {"snes_monitor": None,
    "snes_converged_reason": None,
    "snes_linesearch_type": "bt",
    "snes_linesearch_order": "1",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"}

if obstacle:
    # solve obstacle problem with u >= 0
    sp.update({"snes_type": "vinewtonrsls"})
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="")
    ninf, inf = -1.0e6, 1.0e6
    lower = Function(Z)
    lower.subfunctions[0].dat.data[:] = ninf
    lower.subfunctions[1].dat.data[:] = 0.0
    upper = Function(Z)
    upper.subfunctions[0].dat.data[:] = inf
    upper.subfunctions[1].dat.data[:] = inf
    solver.solve(bounds=(lower, upper))
else:
    # solve Poisson equation without an inequality constraint
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="")
    solver.solve()

sigma, u = sigmau.subfunctions
sigma.rename('sigma')
u.rename('u')
print(f'writing sigma, u to "result.pvd" ...')
VTKFile("result.pvd").write(sigma, u)
