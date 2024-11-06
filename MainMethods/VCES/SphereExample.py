from firedrake import *
try:
    import netgen
except ImportError:
    import sys
    warning("Unable to import NetGen.")
    sys.exit(0)

from firedrake.petsc import PETSc
from netgen.geom2d import SplineGeometry
import numpy as np


tolerance = 1e-10
max_iterations = 8
debug = 1
bracket = [.2, .8]

# Generate initial mesh using netgen
width = 2
TriHeight = .4
geo = SplineGeometry()
geo.AddRectangle(p1=(-1*width, -1*width),
                 p2=(1*width, 1*width),
                 bc="rectangle")

ngmsh = geo.GenerateMesh(maxh=TriHeight)
mesh = Mesh(ngmsh)
mesh.topology_dm.viewFromOptions('-dm_view')
meshHierarchy = [mesh]


def Mark(msh, u, lb, iter, bracket=[.2, .8]):
    # Variable Coefficient Elliptic Smoothing
    # ----------------------------------------------------------------------------------
    W = FunctionSpace(msh, "DG", 0)
    V = FunctionSpace(msh, "CG", 1)
    u0 = Function(V, name="Nodal Active Indicator")
    u1 = Function(V, name="Smoothed Nodal Active Indicator")
    v = TestFunction(V)

    # Compute nodal active set indicator within some tolerance
    NodalDifference = Function(
        V, name="Nodal Difference").interpolate(abs(u - lb))
    NodalActiveIndicator = Function(V).interpolate(
        conditional(NodalDifference < tolerance, 0, 1))

    # Nodal indicator is initial condition to time dependent Heat eq
    u0.assign(NodalActiveIndicator)

    # Vary timestep by average cell circumference of each patch. Applied constant diffusion across all cell sizes
    # (not exactly an average msh.cell_sizes is an L2 projection of the obvious DG0 function into CG1)
    # This forces u and psi to be CG1 unless we work something else out. (like l2 projection into the desired FE space)
    timestep = Function(V)
    timestep.dat.data[:] = 0.5 * msh.cell_sizes.dat.data[:]**2

    # Solve one step implicitly
    F = (inner((u1 - u0)/timestep, v) +
         inner(grad(u1), grad(v))) * dx
    solve(F == 0, u1)

    # Compute average over elements by interpolation into DG0
    u1DG0 = Function(W, name="").interpolate(u1)

    # Applying thresholding parameters
    mark = Function(W, name="Final Marking").interpolate(
        conditional(u1DG0 > bracket[0], conditional(u1DG0 < bracket[1], 1, 0), 0))

    if debug:
        towrite = (u, NodalDifference, u0, u1, u1DG0, mark)

        File(
            'DebugSphaere/VCES[{}]/iterate:{}.pvd'.format(','.join(map(str, bracket)), iter)).write(*towrite)

    return mark


for i in range(max_iterations):

    print("level {}".format(i))
    mesh = meshHierarchy[-1]
    # obstacle and solution are in P1
    V = FunctionSpace(mesh, "CG", 1)
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    # see Chapter 12 of Bueler (2021)
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = - r0 / psi0
    psi_ufl = conditional(le(r, r0), sqrt(1.0 - r * r),
                          psi0 + dpsi0 * (r - r0))
    lb = interpolate(psi_ufl, V)
    # exact solution is known (and it determines Dirichlet boundary)
    afree = 0.697965148223374
    A = 0.680259411891719
    B = 0.471519893402112
    gbdry_ufl = conditional(le(r, afree), psi_ufl, - A * ln(r) + B)
    gbdry = interpolate(gbdry_ufl, V)
    uexact = gbdry.copy()

    # initial iterate is zero
    if i == 0:
        u = Function(V, name="u (FE soln)")
    else:
        # Need to define a destination function space to make cross mesh interpolation work
        V_dest = FunctionSpace(mesh, "CG", 1)
        u = interpolate(u, V_dest)

    # weak form problem; F is residual operator in nonlinear system F==0
    v = TestFunction(V)
    # as in Laplace equation:  - div (grad u) = 0
    F = inner(grad(u), grad(v)) * dx
    bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
    bcs = DirichletBC(V, gbdry, bdry_ids)

    # problem is nonlinear so we need a nonlinear solver, from PETSc's SNES component
    # specific solver is a VI-adapted line search Newton method called "vinewtonrsls"
    # see reference:
    #   S. Benson and T. Munson (2006). Flexible complementarity solvers for large-scale applications,
    #       Optimization Methods and Software, 21, 155–168.
    sp = {"snes_monitor": None,         # prints residual norms for each Newton iteration
          "snes_type": "vinewtonrsls",
          "snes_converged_reason": None,  # prints CONVERGED_... message at end of solve
          "snes_vi_monitor": None,       # prints bounds info for each Newton iteration
          "snes_rtol": 1.0e-8,
          "snes_atol": 1.0e-12,
          "snes_stol": 1.0e-12,
          "snes_vi_zero_tolerance": 1.0e-12,
          "snes_linesearch_type": "basic",
          # these 3 options say Newton step equations are solved by LU
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=sp, options_prefix="")
    ub = interpolate(Constant(PETSc.INFINITY), V)  # no upper obstacle in fact
    # essentially same as:  solve(F == 0, u, bcs=bcs, ...
    solver.solve(bounds=(lb, ub))

    (mark) = Mark(mesh, u, lb, i, bracket)

    nextmesh = mesh.refine_marked_elements(mark)
    meshHierarchy.append(nextmesh)
