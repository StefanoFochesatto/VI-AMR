# Solve the VI problem in section 10.3 of
#   F.-T. Suttmeier (2008).  Numerical Solution of Variational Inequalities
#   by Adaptive Finite Elements, Vieweg + Teubner, Wiesbaden
# Note there is an apparent typo there, since the source f(x,y) needs to be
# negative to generate an active set.

from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
import numpy as np
from viamr import VIAMR

print = PETSc.Sys.Print  # enables correct printing in parallel

refinements = 6
m0 = 10
outfile = "result_suttmeier.pvd"

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

meshhierarchy = [
    UnitSquareMesh(m0, m0),
]
amr = VIAMR()
for i in range(refinements + 1):
    mesh = meshhierarchy[i]
    print(f"solving on mesh {i} ...")
    amr.meshreport(mesh)

    # initial iterate by cross-mesh interpolation from coarser mesh
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u_h").interpolate(Constant(0.0) if i == 0 else u)

    # problem data
    x, y = SpatialCoordinate(mesh)
    psi = Function(V, name="psi").interpolate(
        -(((x - 0.5) ** 2 + (y - 0.5) ** 2) ** (3 / 2))
    )
    # typo? from Suttmeier: f = 10.0 * (x - x**2 + y - y **2)
    fsource = Function(V, name="f").interpolate(-10.0 * (x - x ** 2 + y - y ** 2))

    # weak form and problem
    v = TestFunction(V)
    F = inner(grad(u), grad(v)) * dx - fsource * v * dx
    bcs = DirichletBC(V, Constant(0.0), "on_boundary")
    problem = NonlinearVariationalProblem(F, u, bcs)

    # solve the VI
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=params, options_prefix="s"
    )
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(psi, ub))

    print(f"u_h(1/8,1/4) = {u.at(0.125, 0.25):.6e}")
    if i == refinements:
        break

    # apply VCD AMR, marking inactive by B&R indicator
    #   (choose more refinement in active set, relative to default bracket=[0.2, 0.8])
    mark = amr.vcdmark(mesh, u, psi, bracket=[0.1, 0.8])
    (imark, _, _) = amr.br_mark_poisson(u, psi, f=fsource)
    # imark = amr.eleminactive(u, psi)  # alternative is to refine all inactive
    _, DG0 = amr.spaces(mesh)
    mark = Function(DG0).interpolate((mark + imark) - (mark * imark))  # union
    mesh = amr.refinemarkedelements(mesh, mark)
    meshhierarchy.append(mesh)

print(f"done ... writing u_h, psi, f, gap=u_h-psi to {outfile} ...")
gap = Function(V, name="gap = u_h - psi").interpolate(u - psi)
VTKFile(outfile).write(u, psi, fsource, gap)
