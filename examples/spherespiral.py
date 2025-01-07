'''Run UDO and VCES methods on two obstacle problems.
"sphere" problem has exact solution, "spiral" does not.
Generates 4 .pvd output files.'''

import numpy as np
from firedrake import *
from viamr import VIAMR

try:
    import netgen
except ImportError:
    print("ImportError.  Unable to import NetGen.  Exiting.")
    import sys
    sys.exit(0)
from netgen.geom2d import SplineGeometry

refinements = 3
debugoutputs = True

problems = ['sphere', 'spiral']
widths = [2.0, 1.0]
TriHeights = [0.4, 0.1]


def get_mesh(width, TriHeight):
    # re-generate initial mesh using netgen
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-1 * width, -1 * width),
                     p2=(1 * width, 1 * width), bc="rectangle")
    ngmsh = None
    ngmsh = geo.GenerateMesh(maxh=TriHeight)
    return Mesh(ngmsh)


def sphere_problem(mesh, V):
    # exactly-solvable obstacle problem for spherical obstacle
    # see Chapter 12 of Bueler (2021)
    # obstacle and solution are in function space V
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    psi_ufl = conditional(le(r, r0), sqrt(
        1.0 - r * r), psi0 + dpsi0 * (r - r0))
    psi = Function(V).interpolate(psi_ufl)
    # exact solution is known (and it determines Dirichlet boundary)
    afree = 0.697965148223374
    A = 0.680259411891719
    B = 0.471519893402112
    gbdry_ufl = conditional(le(r, afree), psi_ufl, -A * ln(r) + B)
    gbdry = Function(V).interpolate(gbdry_ufl)
    uexact = gbdry
    return psi, gbdry, uexact


def spiral_problem(mesh, V):
    # spiral obstacle problem from section 7.1.1 in Graeser & Kornhuber (2009)
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    theta = atan2(y, x)
    tmp = sin(2.0*pi/r + pi/2.0 - theta) + r * \
        (r+1) / (r - 2.0) - 3.0 * r + 3.6
    psi = Function(V).interpolate(conditional(le(r, 1.0e-8), 3.6, tmp))
    gbdry = Constant(0.0)
    uexact = None  # not known
    return psi, gbdry, uexact


for prob in range(2):
    for method in ['udo', 'vces']:
        mesh = get_mesh(widths[prob], TriHeights[prob])
        for i in range(refinements + 1):
            # get initial iterate from previous mesh (or zero when i==0)
            print(f"{problems[prob]} {method} level {i}")
            V = FunctionSpace(mesh, "CG", 1)
            if i == 0:
                u = Function(V)  # set to zero
            else:
                # cross-mesh interpolation
                u = Function(V).interpolate(u)
            u.rename("solution u")
            VIAMR().meshreport(mesh)

            # set up weak form problem; F is residual operator in nonlinear system F==0
            if prob == 0:
                lbound, gbdry, uexact = sphere_problem(mesh, V)
            else:
                lbound, gbdry, uexact = spiral_problem(mesh, V)
            lbound.rename("obstacle psi")
            v = TestFunction(V)
            # interior condition of VI is Laplace equation  - div (grad u) = 0
            F = inner(grad(u), grad(v)) * dx
            bdry_ids = (1, 2, 3, 4)  # all four sides of boundary
            bcs = DirichletBC(V, gbdry, bdry_ids)

            # set up nonlinear solver
            # Note that VI problems are inherently nonlinear.  This solver is from PETSc's
            # SNES component, a VI-adapted line search Newton method called "vinewtonrsls"
            # (Benson & Munson, 2006).
            sp = {
                "snes_type": "vinewtonrsls",
                "snes_converged_reason": None,
                # "snes_monitor": None,
                # "snes_vi_monitor": None, # prints bounds info for each Newton iteration
                "snes_rtol": 1.0e-8,
                "snes_atol": 1.0e-12,
                "snes_stol": 0.0,
                "snes_vi_zero_tolerance": 1.0e-12,
                "snes_linesearch_type": "basic",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
            problem = NonlinearVariationalProblem(F, u, bcs)

            # solve VI problem
            solver = NonlinearVariationalSolver(
                problem, solver_parameters=sp, options_prefix=""
            )
            ubound = Function(V).interpolate(
                Constant(PETSc.INFINITY))  # no upper obstacle
            solver.solve(bounds=(lbound, ubound))
            if uexact is not None:
                print(f"  L2 error norm = {errornorm(u, uexact):.3e}")
            if i == refinements:
                break

            # refine via marking
            if method == 'udo':
                mark = VIAMR().udomark(mesh, u, lbound)
            else:
                mark = VIAMR().vcesmark(mesh, u, lbound, bracket=[
                    0.06, 0.94])  # more refinement
            if debugoutputs:
                VTKFile(f"result_{problems[prob]}_{method}{i}.pvd").write(
                    u, lbound, mark)

            # use NetGen to get next mesh
            mesh = mesh.refine_marked_elements(mark)

        # write results
        ugap = Function(V).interpolate(u - lbound)
        ugap.rename('gap u - psi')
        if uexact is not None:
            uerror = Function(V, name="u error").interpolate(abs(u - uexact))
            VTKFile(f"result_{problems[prob]}_{method}.pvd").write(
                u, lbound, ugap, uerror)
        else:
            VTKFile(f"result_{problems[prob]}_{method}.pvd").write(
                u, lbound, ugap)
