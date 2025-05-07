# solve Example 1 from
#   M. Ainsworth, J. T. Oden, & C. Y. Lee (1993).  Local a posteriori error estimators
#   for variational inequalities, Numer. Methods for Partial Diff. Eqn. 9, 23--33
# it has a simple exact solution

from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
import numpy as np
from viamr import VIAMR

print = PETSc.Sys.Print  # enables correct printing in parallel

refinements = 6
m_initial = 10
outfile = "result_aol1.pvd"
ifracexact = np.pi/2 - 1.0

params = {
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-12,
    "snes_linesearch_type": "basic",
    # "snes_monitor": None,
    #"snes_vi_monitor": None,
    "snes_converged_reason": None,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-12,
    "snes_stol": 1.0e-12,
    "snes_max_it": 200,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

initial_mesh = RectangleMesh(m_initial, 2 * m_initial, 0.5, 1.0)

amr = VIAMR(activetol=1.0e-12)
meshhierarchy = [
    initial_mesh,
]
for i in range(refinements + 1):
    mesh = meshhierarchy[i]
    print(f"solving on mesh {i} ...")
    amr.meshreport(mesh)

    # exact solution as Dirichlet boundary data
    # see: page 31 of Ainsworth, Oden, Lee (1993)
    x, y = SpatialCoordinate(mesh)
    rr = (x + 1) ** 2 + y ** 2
    g_ufl = conditional(lt(rr, 2.0), 0.25 * rr - 0.5 - 0.5 * ln(0.5 * rr), 0.0)

    # problem and weak form
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u_h")
    v = TestFunction(V)
    uexact = Function(V, name="u_exact").interpolate(g_ufl)
    F = inner(grad(u), grad(v)) * dx - Constant(-1.0) * v * dx
    bcs = DirichletBC(V, uexact, (1, 2, 3, 4))
    problem = NonlinearVariationalProblem(F, u, bcs)

    # solve the VI
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=params, options_prefix="s"
    )
    lb = Function(V).interpolate(Constant(0.0))
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(lb, ub))
    print(f"  |u - u_exact|_2 = {errornorm(u, uexact):.3e}")

    # evaluate inactive fraction
    ifrac = assemble(amr.eleminactive(u, lb) * dx) / 0.5  # domain area = 0.5
    print(f"  computed inactive fraction {ifrac:.6f} (vs {ifracexact:.6f})")

    # apply VCD AMR, and mark inactive by B&R indicator
    if i == refinements:
        break
    mark = amr.vcdmark(mesh, u, lb)
    imark, _, _ = amr.br_mark_poisson(u, lb, f=Constant(-1.0), theta=0.9)
    #imark = amr.eleminactive(u, lb)
    _, DG0 = amr.spaces(mesh)
    mark = Function(DG0).interpolate((mark + imark) - (mark * imark))  # union
    mesh = amr.refinemarkedelements(mesh, mark)
    meshhierarchy.append(mesh)

print(f"done ... writing u_h, u_exact, error=|u_h-u_exact| to {outfile} ...")
error = Function(V, name="error = |u - u_exact|")
error.interpolate(abs(u - uexact))
ielem = amr.eleminactive(u, lb)
VTKFile(outfile).write(u, uexact, error, ielem)
