from firedrake import *

# Setting up the Mesh.
m = 20  # regular m x m mesh
# set up mesh on square
width, offset = 4.0, -2.0   # Omega = (-2,2)^2
mesh = SquareMesh(m, m, width, quadrilateral=False)
# Adding offset to center mesh
mesh.coordinates.dat.data[:, :] += offset

# Get mesh information
mesh.topology_dm.viewFromOptions('-dm_view')

# We'll define our obstacle and solution space to be CG1
V = FunctionSpace(mesh, "CG", 1)

# Define the obstacle function psi
(x, y) = SpatialCoordinate(mesh)
r = sqrt(x * x + y * y)
r0 = 0.9
psi0 = np.sqrt(1.0 - r0 * r0)
dpsi0 = - r0 / psi0
psi_ufl = conditional(le(r, r0), sqrt(1.0 - r * r),
                      psi0 + dpsi0 * (r - r0))
lb = Function(V).interpolate(psi_ufl)

# exact solution is known (and it determines Dirichlet boundary)
afree = 0.697965148223374
A = 0.680259411891719
B = 0.471519893402112
gbdry_ufl = conditional(le(r, afree), psi_ufl, - A * ln(r) + B)
gbdry = Function(V).interpolate(gbdry_ufl)
uexact = gbdry.copy()

# initial iterate is zero
u = Function(V, name="u (FE soln)")

# weak form problem; F is residual operator in nonlinear system F==0
v = TestFunction(V)
F = inner(grad(u), grad(v)) * dx
bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
bcs = DirichletBC(V, gbdry, bdry_ids)

# problem is nonlinear so we need a nonlinear solver, from PETSc's SNES component
# specific solver is a VI-adapted line search Newton method called "vinewtonrsls"

sp = {"snes_vi_monitor": None,     # prints residual norms & active set nodes
      "snes_type": "vinewtonrsls",  # VI Newton with with reduced space line search
      "snes_converged_reason": None,  # prints CONVERGED_... message at end of solve
      "snes_rtol": 1.0e-8,  # Relative convergence tolerance
      # Absolute convergence tolerance (Applied to the residual of basis functions on the inactive set)
      "snes_atol": 1.0e-12,
      "snes_stol": 1.0e-12,  # Norm of step convergence tolerance
      "snes_vi_zero_tolerance": 1.0e-12,
      "snes_linesearch_type": "basic",
      # these 3 options say Newton step equations are solved by LU
      "ksp_type": "preonly",
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps"}
problem = NonlinearVariationalProblem(F, u, bcs)
solver = NonlinearVariationalSolver(
    problem, solver_parameters=sp, options_prefix="")
ub = Function(V).interpolate(Constant(PETSc.INFINITY))  # no upper obstacle
solver.solve(bounds=(lb, ub))
