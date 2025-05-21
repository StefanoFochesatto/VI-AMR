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
import argparse
print = PETSc.Sys.Print  # enables correct printing in parallel


parser = argparse.ArgumentParser(description="Solve VI problem with AMR.")
parser.add_argument('--total', action='store_true',
                    help='Enable total marking strategy (default: False)')
parser.add_argument('--theta', type=float, default=0.5,
                    help='Fraction of elements to mark for refinement (default: 0.5)')
parser.add_argument('--refinements', type=int, default=4,
                    help='Number of refinements (default: 4)')
parser.add_argument('--m0', type=int, default=10,
                    help='initial mesh subdivision (default: 10)')

args = parser.parse_args()

refinements = args.refinements 
m0 = args.m0 

if args.total:
    outfile = "result_suttmeier_total.pvd"
else:
    outfile = "result_suttmeier_max.pvd"

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

# explicitly setting distribution parameters allows this to be a udomark() example
# which still runs in parallel
meshhierarchy = [
    UnitSquareMesh(
        m0,
        m0,
        diagonal="crossed",
        distribution_parameters={
            "partition": True,
            "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
        },
    ),
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
    fsource = Function(V, name="f").interpolate(-10.0 * (x - x**2 + y - y**2))

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

    mark = amr.udomark(u, psi, n=2)

    # alternative: apply VCD AMR, marking inactive by B&R indicator
    #   (choose more refinement in active set, relative to default bracket=[0.2, 0.8])
    # mark = amr.vcdmark(u, psi, bracket=[0.2, 0.9])

    residual = -div(grad(u)) - fsource
    
    (imark, _, _) = amr.brinactivemark(u, psi, residual, theta =  args.theta, total = args.total)
    # imark = amr.eleminactive(u, psi)  # alternative is to refine all inactive
    mark = amr.unionmarks(mark, imark)
    mesh = amr.refinemarkedelements(mesh, mark)
    meshhierarchy.append(mesh)

print(f"done ... writing u_h, psi, f, gap=u_h-psi to {outfile} ...")
gap = Function(V, name="gap = u_h - psi").interpolate(u - psi)
VTKFile(outfile).write(u, psi, fsource, gap)
