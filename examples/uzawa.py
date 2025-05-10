# Apply the dual (Lagrangian) formulation from section 10.1 of
#   F.-T. Suttmeier (2008).  Numerical Solution of Variational Inequalities
#   by Adaptive Finite Elements, Vieweg + Teubner, Wiesbaden
# to the sphere problem, and then apply the (Suttmeier recommended?) Uzawa
# algorithm, which solves a sequence of linear PDEs using an iteration
# on the Lagrange multiplier lambda.
#
# This is *not* a strong VI solver algorithm.  It does not converge
# quadratically like a Newton iteration.  Also, it does unnecessary work
# in the active set.  However, one can solve a VI without a VI solver!

from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
import numpy as np
from viamr import VIAMR

print = PETSc.Sys.Print  # enables correct printing in parallel

refinements = 4
m0 = 10
outfile = "result_uzawa.pvd"

uzawa_rho = 2.0  # *not clear* what this value should be, as it
#   mixes units of displacement and residual
uzawa_iter = 6  # minimum iteration count ... ad hoc increase on levels (below)

params = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

mesh0 = RectangleMesh(m0, m0, 2.0, 2.0, originX=-2.0, originY=-2.0, diagonal="crossed")
meshhierarchy = [
    mesh0,
]
amr = VIAMR(debug=False)  # turn off checking that u >= psi
for i in range(refinements + 1):
    mesh = meshhierarchy[i]
    print(f"solving on mesh {i} ...")
    amr.meshreport(mesh)

    # CG1 for u, DG0 for Lagrange multiplier lambda
    V = FunctionSpace(mesh, "CG", 1)
    DG0 = FunctionSpace(mesh, "DG", 0)

    # lambda from Uzawa step; includes cross-mesh interpolation
    if i == 0:
        lam = Function(DG0)
    else:
        newlam = Function(DG0).interpolate(lam + uzawa_rho * (psi - u))
        lam = Function(DG0).interpolate(max_value(Constant(0.0), newlam))
    lam.rename("lambda")

    # problem data on this mesh
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    psi_ufl = conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))
    psi = Function(V, name="psi").interpolate(psi_ufl)
    afree, A, B = 0.697965148223374, 0.680259411891719, 0.471519893402112
    uex_ufl = conditional(le(r, afree), psi, -A * ln(r) + B)
    uexact = Function(V, name="u_exact").interpolate(uex_ufl)

    # weak form and problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = lam * v * dx
    bcs = [
        DirichletBC(V, uexact, "on_boundary"),
    ]

    # solve the *linear PDE* iteratively
    u = Function(V, name="u_h")
    for k in range(2 * i + uzawa_iter):  # more iterations on each level
        solve(a == L, u, bcs=bcs, solver_parameters=params, options_prefix="s")
        print(f"  |u - u_exact|_2 = {errornorm(u, uexact):.3e}")
        delta = Function(DG0).interpolate(psi - u)
        lam.interpolate(max_value(Constant(0.0), lam + uzawa_rho * delta))

    if i == refinements:
        break

    # apply VCD AMR, marking inactive by B&R indicator
    mark = amr.vcdmark(mesh, u, psi)
    residual = -div(grad(u))
    (imark, _, _) = amr.brinactivemark(u, psi, residual)
    mark = amr.unionmarks(mark, imark)
    mesh = amr.refinemarkedelements(mesh, mark)
    meshhierarchy.append(mesh)

print(f"done ...")
print(
    f"writing u_h, psi, max(u_h,psi), lam, gap=u_h-psi, uexact, error to {outfile} ..."
)
uhadmiss = Function(V, name="u_h (admissible)").interpolate(max_value(u, psi))
error = Function(V, name="error = |u - u_exact|").interpolate(abs(u - uexact))
gap = Function(V, name="gap = u_h (admiss) - psi").interpolate(uhadmiss - psi)
VTKFile(outfile).write(u, psi, uhadmiss, lam, gap, uexact, error)
