# Apply the dual (Lagrangian) method from section 10.1 of
#   F.-T. Suttmeier (2008).  Numerical Solution of Variational Inequalities
#   by Adaptive Finite Elements, Vieweg + Teubner, Wiesbaden
# to the sphere problem.

from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
import numpy as np
from viamr import VIAMR

print = PETSc.Sys.Print  # enables correct printing in parallel

refinements = 1
m0 = 10
outfile = "result_dualmixed.pvd"

params = {
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-12,
    "snes_linesearch_type": "basic",
    # "snes_monitor": None,
    # "snes_vi_monitor": None,
    "snes_converged_reason": None,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-12,
    "snes_stol": 1.0e-12,
    "snes_max_it": 200,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

mesh0 = RectangleMesh(m0, m0, 2.0, 2.0, originX=-2.0, originY=-2.0)
meshhierarchy = [mesh0, ]
amr = VIAMR()
for i in range(refinements + 1):
    mesh = meshhierarchy[i]
    print(f"solving on mesh {i} ...")
    amr.meshreport(mesh)

    # mixed space, with DG0 for Lagrange multiplier
    V = FunctionSpace(mesh, "CG", 1)
    DG0 = FunctionSpace(mesh, "DG", 0)
    Z = V * DG0

    # initial iterate by cross-mesh interpolation from coarser mesh
    if i == 0:
        ul = Function(Z)
    else:
        ul = Function(Z).interpolate(ul)
        #VTKFile("tmp.pvd").write(ul.subfunctions[0], ul.subfunctions[1])

    # problem data
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

    if i > 0:
        ul.subfunctions[0].interpolate(max_value(ul.subfunctions[0], psi))

    # weak form and problem
    u, lam = split(ul)
    phi, w = TestFunctions(Z)
    F = inner(grad(u), grad(phi)) * dx - lam * phi * dx + (u - psi) * w * dx
    bcs = [DirichletBC(Z.sub(0), uexact, "on_boundary"), ]
    problem = NonlinearVariationalProblem(F, ul, bcs=bcs)

    # mixed-space bounds
    ninf, inf = -1.0e6, 1.0e6
    lower = Function(Z)
    lower.subfunctions[0].dat.data[:] = ninf
    lower.subfunctions[1].dat.data[:] = 0.0
    upper = Function(Z)
    upper.subfunctions[0].dat.data[:] = inf
    upper.subfunctions[1].dat.data[:] = inf

    # solve the VI
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=params, options_prefix="s"
    )
    solver.solve(bounds=(lower, upper))

    if i == refinements:
        break

    # apply VCD AMR, marking inactive by B&R indicator
    u = ul.subfunctions[0]
    mark = amr.vcdmark(mesh, u, psi)
    (imark, _, _) = amr.br_mark_poisson(u, psi)
    # imark = amr.eleminactive(u, psi)  # alternative is to refine all inactive
    _, DG0 = amr.spaces(mesh)
    mark = Function(DG0).interpolate((mark + imark) - (mark * imark))  # union
    mesh = amr.refinemarkedelements(mesh, mark)
    meshhierarchy.append(mesh)

print(f"done ... writing u_h, psi, lam, gap=u_h-psi, uexact to {outfile} ...")
u, lam = ul.subfunctions
u.rename("u_h")
lam.rename("lambda")
gap = Function(V, name="gap = u_h - psi").interpolate(u - psi)
VTKFile(outfile).write(u, psi, lam, gap, uexact)
