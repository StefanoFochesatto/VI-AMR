import argparse

parser = argparse.ArgumentParser(
    description="""Solve the VI problem in section 10.3 of
   F.-T. Suttmeier (2008).  Numerical Solution of Variational Inequalities
   by Adaptive Finite Elements, Vieweg + Teubner, Wiesbaden
Note there is an apparent typo there, since the source f(x,y) needs to be
negative to generate an active set.  When run in serial with -samplepoint
we report the solution value u_h(1/8,1/4), for comparison to the reference."""
)
parser.add_argument(
    "-samplepoint",
    action="store_true",
    help="print u_h(1/8,1/4) (default: False; serial only)",
)
parser.add_argument(
    "-m0", type=int, default=10, help="initial mesh subdivision (default: 10)"
)
parser.add_argument(
    "-opvd",
    metavar="FILE",
    type=str,
    default="",
    help="output file name for Paraview format (.pvd)",
)
parser.add_argument(
    "-refinements", type=int, default=2, help="number of refinements (default: 2)"
)
parser.add_argument(
    "-theta",
    type=float,
    default=0.5,
    help="fraction of elements to mark for refinement (default: 0.5)",
)
parser.add_argument(
    "-total",
    action="store_true",
    help="enable total marking strategy (default: False)",
)

args, passthroughoptions = parser.parse_known_args()

import petsc4py

petsc4py.init(passthroughoptions)

import numpy as np
from firedrake import *
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel
from viamr import VIAMR

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
    UnitSquareMesh(
        args.m0,
        args.m0,
        diagonal="crossed",
        # FIXME explicitly setting distribution parameters allows udomark() to run in parallel
        distribution_parameters={
            "partition": True,
            "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
        },
    ),
]
amr = VIAMR()
for i in range(args.refinements + 1):
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

    # protect this print from the issue
    #   https://www.firedrakeproject.org/point-evaluation.html#evaluation-with-a-distributed-mesh
    if args.samplepoint and mesh.comm.size == 1:
        print(f"u_h(1/8,1/4) = {u.at(0.125, 0.25):.6e}")
    if i == args.refinements:
        break

    # mark by default UDO and BR in inactive region
    fbmark = amr.udomark(u, psi)
    residual = -div(grad(u)) - fsource
    (imark, _, _) = amr.brinactivemark(
        u, psi, residual, theta=args.theta, method="total" if args.total else "max"
    )
    mark = amr.unionmarks(fbmark, imark)
    mesh = amr.refinemarkedelements(mesh, mark)
    meshhierarchy.append(mesh)

print("done ...")
if args.opvd:
    print(f"writing u_h, psi, f, gap=u_h-psi, rank to {args.opvd} ...")
    gap = Function(V, name="gap = u_h - psi").interpolate(u - psi)
    rank = Function(FunctionSpace(mesh, "DG", 0))
    rank.dat.data[:] = mesh.comm.rank
    rank.rename("rank")
    VTKFile(args.opvd).write(u, psi, fsource, gap, rank)
