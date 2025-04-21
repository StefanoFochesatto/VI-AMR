from firedrake import *
import matplotlib.pyplot as plt
import numpy as np


# Exact free boundary are approximately = \pm 0.2928932188134524

# different mesh sequences give different results because errors
# are dominated by the gap (see below), which is a discontinuous
# function of h
# MeshSizes = [10, 20, 40, 100]
# MeshSizes = [10, 20, 40, 100, 200]
# MeshSizes = [10, 20, 40, 100, 200, 400]
MeshSizes = [10, 20, 40, 100, 200, 400, 1000]  # the worst
# MeshSizes = [10, 20, 40, 100, 200, 400, 1000, 2000]
# MeshSizes = [10, 20, 40, 100, 200, 400, 1000, 2000, 4000]


# Initializing lists to store results
HistError_LInf = []
HistError_L2 = []
HistError_H1 = []
HistGap = []
h = [2.0 / i for i in MeshSizes]

# Exact Free Boundary Points
b = (2 - (2)**.5) * .5
a = -(2 - (2)**.5) * .5
# print(b)

for i in MeshSizes:
    # Initialize Mesh
    mesh = IntervalMesh(i, -1, 1)

    # Initialize Function Space
    V = FunctionSpace(mesh, 'CG', 1)

    # Define Dirichlet Boundary Conditions and Obstacle
    x = SpatialCoordinate(mesh)[0]  # <- use [0] in one-dimensional case
    psi_ufl = .5 - x**2
    # The discrete solution is not continuum admissible with this example, because interpolating the continuum obstacle 
    # gives a discrete obstacle is lower. 
    # Maybe replace this with R+ operator in FASCD
    lb = Function(V).interpolate(psi_ufl)
    gbdry = Function(V).interpolate(Constant(0.0))
    bdry_ids = (1, 2)
    bcs = DirichletBC(V, gbdry, bdry_ids)

    # Define Weak Form
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v)) * dx

    # Define Solver Parameters and Solve with VINEWTONRSLS
    sp = {  # "snes_monitor": None,
        "snes_type": "vinewtonrsls",
        "snes_converged_reason": None,
        "snes_max_it": 1000,
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-12,
        "snes_stol": 1.0e-12,
        "snes_vi_zero_tolerance": 1.0e-12,
        "snes_linesearch_type": "basic",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=sp, options_prefix="")
    # no upper obstacle in fact
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(lb, ub))

    # Generating UFL for Exact Solution
    leftline = ((.5 - a**2)/(a + 1))*(x + 1)
    rightline = -((.5 - b**2)/(1 - b))*(x - 1)
    u_exact_ufl = conditional(le(x, a), leftline, conditional(
        ge(x, b), rightline, psi_ufl))

    uerr = Function(V).interpolate(u_exact_ufl - u)  # NOT VExact here

    # Computing Error Norms
    error_LInf = np.max(np.abs(uerr.dat.data))
    error_L2 = norm(uerr)
    error_H1 = norm(uerr, 'H1')

    HistError_LInf.append(error_LInf)
    HistError_L2.append(error_L2)
    HistError_H1.append(error_H1)
    HistGap.append(
        np.min(np.abs(mesh.topology_dm.getCoordinatesLocal().array - b)))

p_L2 = np.polyfit(np.log(h), np.log(HistError_L2), 1)
p_H1 = np.polyfit(np.log(h), np.log(HistError_H1), 1)

print('Convergence Rate L2 = ', p_L2[0])
print('Convergence Rate H1 = ', p_H1[0])

plt.loglog(h, HistError_L2, 'o', color='C2',
           label=f'L2 Error O(h^{p_L2[0]:.2f})')
plt.loglog(h, np.exp(p_L2[0] * np.log(h) + p_L2[1]), '--', color='C2')
plt.loglog(h, HistError_H1, 'o', color='C3',
           label=f'H1 Error O(h^{p_H1[0]:.2f})')
plt.loglog(h, np.exp(p_H1[0] * np.log(h) + p_H1[1]), '--', color='C3')
plt.loglog(h, HistGap, 'o', color='C4', label='gap = min |x_j - b|')
plt.xlabel('h')
plt.gca().set_xlim([0.001, 1.0])
plt.ylabel('Error')
plt.legend()
plt.show()
