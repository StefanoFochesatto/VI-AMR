from argparse import ArgumentParser, RawTextHelpFormatter

parser = ArgumentParser(
    description="""
Solves a 2D steady, isothermal shallow ice approximation glacier obstacle problem.

The default set-up is a synthetic problem over a square domain [0,L]^2 with
L = 1800.0 km.

A data-based problem (-data DATA.nc) is a WIP.

By default (-prob dome) we solve a flat bed case where the exact solution is
known.  Option -prob cap generates a random, but smooth, bed topography, but
keeps the dome SMB.  Option -prob range generates a different SMB and a
disconnected glacier.

We apply the UDO or VCD methods for free-boundary refinement.  The default mode
does n=1 UDO at the free boundary plus gradient-recovery refinement in the
inactive set.  The default marking strategy in the inactive set uses the "total"
fixed-rate strategy.

The default VI solver is Picard iteration on the tilt (Jouvet & Bueler, 2012),
and vinewtonrsls (+ mumps) for each tilt.  A full Newton iteration, i.e. simply
vinewtonrsls, is also available.

Examples:
  python3 steady.py -opvd dome.pvd
  mpiexec -n 4 python3 steady.py -refine 4 -prob range -opvd range.pvd

Excellent -prob dome convergence:
  mpiexec -n 4 python3 steady.py -refine 8 -newton

High-resolution example; achieved 30 m resolution along ice sheet margin:
FIXME: redo now that "total" fixed-rate and diagonal="crossed" are the defaults
  mpiexec -n 20 python3 steady.py -prob range -m 50 -refine 10 -opvd result_range.pvd
(Large memory needed ...)  Note L=1800 km so h_min / L ~ 1e-5.  However,
such runs reveal less than perfect refinement right along the free boundary
at very high resolution.

The -newton solve works well but needs a coarser initial grid:
  mpiexec -n 20 python3 steady.py -prob range -newton -m 5 -refine 10 -rmethod alternate -opvd result_rangenewton.pvd

I don't yet trust this kind of -data run, but it converges o.k.:
  python3 steady.py -data eastgr.nc -opvd result_data.pvd -refine 5 -rmethod alternate -freezecount 24
""",
    formatter_class=RawTextHelpFormatter,
)
parser.add_argument(
    "-data",
    metavar="FILE",
    type=str,
    default="",
    help='read "topg" variable from NetCDF file (.nc)',
)
parser.add_argument(
    "-freezecount",
    type=int,
    default=8,
    metavar="P",
    help="number of Picard frozen-tilt iterations [default=8]",
)
parser.add_argument(
    "-m",
    type=int,
    default=10,
    metavar="M",
    help="number of cells in each direction on initial mesh [default=10]",
)
parser.add_argument(
    "-newton",
    action="store_true",
    default=False,
    help="use straight Newton instead of Picard+Newton; not robust",
)
parser.add_argument(
    "-opvd",
    metavar="FILE",
    type=str,
    default="",
    help="output file name for Paraview format (.pvd)",
)
parser.add_argument(
    "-prob",
    type=str,
    default="dome",
    metavar="X",
    choices=["dome", "cap", "range"],
    help="choose problem from {dome, cap, range}",
)
parser.add_argument(
    "-refine",
    type=int,
    default=3,
    metavar="R",
    help="number of AMR refinements [default 3]",
)
parser.add_argument(
    "-theta",
    type=float,
    default=0.3,
    metavar="X",
    help="theta to use in 'total' fixed-rate marking strategy in inactive set [default=0.3]",
)
parser.add_argument(
    "-vcd",
    action="store_true",
    default=False,
    help="apply VCD free-boundary marking (instead of UDO)",
)
args, passthroughoptions = parser.parse_known_args()

import numpy as np
import petsc4py

petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.petsc import PETSc

print = PETSc.Sys.Print
from viamr import VIAMR

from synthetic import secpera, n, Gamma, L, dome_exact, accumulation, bumps

assert args.m >= 1, "at least one cell in mesh"
assert args.refine >= 0, "cannot refine a negative number of times"

# read data for bed topography
if args.data:
    print("ignoring -prob choice ...")
    args.prob = None
    print(f"reading topg from NetCDF file {args.data} with native data grid:")
    from datanetcdf import DataNetCDF

    topg_nc = DataNetCDF(args.data, "topg")
    topg_nc.preview()
    topg_nc.describe_grid(print=PETSc.Sys.Print, indent=4)
    print(f"putting topg onto matching Firedrake structured data mesh ...")
    topg, nearb = topg_nc.function(delnear=100.0e3)
else:
    print(
        f"generating synthetic {args.m} x {args.m} initial mesh for problem {args.prob} ..."
    )

# setting distribution parameters should not be necessary ...
dp = {
    "partition": True,
    "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
}
if args.data:
    # generate netgen mesh compatible with data mesh, but unstructured
    # and at user (-m) resolution, typically lower
    mesh = topg_nc.ngmesh(args.m, distribution_parameters=dp)
else:
    # generate [0,L]^2 mesh via Firedrake
    mesh = RectangleMesh(
        args.m, args.m, L, L, diagonal="crossed", distribution_parameters=dp
    )

# solver parameters
sp = {
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-2,  # max u ~ 10^9, so roughly within 1 part in 10^-11 for u=H^{8/3}
    "snes_rtol": 1.0e-6,
    "snes_atol": 1.0e-10,
    "snes_stol": 1.0e-10,  # FIXME??  why does it even matter?  in any case, keep it tight
    # "snes_monitor": None,
    # "snes_vi_monitor": None,
    "snes_converged_reason": None,
    # "snes_linesearch_type": "basic",
    "snes_linesearch_type": "bt",
    "snes_linesearch_order": "1",
    "snes_max_it": 1000,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# transformed SIA
p = n + 1  # typical:  p = 4
omega = (p - 1) / (2 * p)  #  omega = 3/8
phi = (p + 1) / (2 * p)  #  phi = 5/8
r = p / (p - 1)  #  r = 4/3


def Phi(u, b):
    return -(1.0 / omega) * (u + 1.0) ** phi * grad(b)  # eps=1 regularization is small


def weakform(u, a, b, Z=None):
    v = TestFunction(u.function_space())
    if Z is not None:
        du_tilt = grad(u) - Z
    else:
        du_tilt = grad(u) - Phi(u, b)
    Dp = inner(du_tilt, du_tilt) ** ((p - 2) / 2)
    return Gamma * omega ** (p - 1) * Dp * inner(du_tilt, grad(v)) * dx - a * v * dx


# outer mesh refinement loop
amr = VIAMR(debug=True)
for i in range(args.refine + 1):
    # mark and refine based on constraint u >= 0
    if i > 0:
        print(f"refining free boundary ({'VCD' if args.vcd else 'UDO'})", end="")
        if args.vcd:
            # change bracket vs default [0.2, 0.8], to provide more high-res
            #   for ice near margin (0.2 -> 0.1), i.e. on inactive side
            fbmark = amr.vcdmark(u, lb, bracket=[0.1, 0.8])
        else:
            fbmark = amr.udomark(u, lb, n=1)
        print(" and by gradient recovery in inactive ...")
        imark, _, _ = amr.gradrecinactivemark(u, lb, theta=args.theta, method="total")
        mark = amr.unionmarks(fbmark, imark)
        mesh = amr.refinemarkedelements(mesh, mark)

    # describe current mesh
    nv, ne, hmin, hmax = amr.meshsizes(mesh)
    print(f"solving problem {args.prob} on mesh level {i}:")
    amr.meshreport(mesh)

    # space for most functions
    V = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)

    # surface mass balance function on current mesh
    if args.data:
        # SMB from linear model based on lapse rate; from linearizing dome case
        c0 = -3.4e-8
        c1 = (6.3e-8 - c0) / 3.6e3
        a_lapse = c0 + c1 * topg
        a = Function(V).interpolate(
            conditional(nearb > 0.0, -1.0e-6, a_lapse)
        )  # also cross-mesh re nearb
    else:
        a = Function(V).interpolate(accumulation(x, problem=args.prob))
    a.rename("a = accumulation")

    # bedrock on current mesh
    if args.data:
        b = Function(V).project(topg)  # cross-mesh projection from data mesh
    else:
        if args.prob == "dome":
            b = Function(V).interpolate(Constant(0.0))
            sexact = Function(V).interpolate(dome_exact(x))
            sexact.rename("s_exact")
        else:
            b = Function(V).interpolate(bumps(x, problem=args.prob))
    b.rename("b = bedrock topography")

    # exact solution on current mesh if available
    if not args.data and args.prob == "dome":
        sexact = Function(V).interpolate(dome_exact(x))
        sexact.rename("s_exact")

    # initialize transformed thickness variable
    if i == 0:
        # build pile of ice from accumulation
        pileage = 400.0  # years
        Hinit = pileage * secpera * conditional(a > 0.0, a, 0.0)
        uold = Function(V).interpolate(Hinit ** omega)
    else:
        # cross-mesh interpolation of previous solution
        uold = Function(V).interpolate(u)
        # remove sign flaws from cross-mesh interpolation
        #   note: u = H^(8/3) < 1 is *very* little ice in an initial iterate
        uold = Function(V).interpolate(conditional(uold < 1.0, 0.0, uold))

    # solve on current mesh
    u = Function(V, name="u = transformed thickness").interpolate(uold)
    lb = Function(V).interpolate(Constant(0.0))  # lower bound *in solver*
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    bcs = [
        DirichletBC(V, Constant(0.0), "on_boundary"),
    ]
    if args.newton:
        F = weakform(u, a, b)
        problem = NonlinearVariationalProblem(F, u, bcs=bcs)
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="s"
        )
        solver.solve(bounds=(lb, ub))
    else:
        # outer loop for freeze iteration
        for k in range(args.freezecount):
            # print(f'  freeze tilt iteration {k+1} ...')
            Ztilt = Phi(uold, b)
            F = weakform(u, a, b, Z=Ztilt)
            problem = NonlinearVariationalProblem(F, u, bcs=bcs)
            solver = NonlinearVariationalSolver(
                problem, solver_parameters=sp, options_prefix="s"
            )
            solver.solve(bounds=(lb, ub))
            uold = Function(V).interpolate(u)

    # update true geometry variables
    H = Function(V, name="H = thickness").interpolate(u ** omega)
    s = Function(V, name="s = surface elevation").interpolate(b + H)

    # report numerical errors if exact solution known
    if args.prob == "dome":
        sdiff = Function(V).interpolate(s - dome_exact(x))
        sdiff.rename("sdiff = s - s_exact")
        err_l2 = norm(sdiff / L)
        err_av = norm(sdiff, "l1") / L ** 2
        print("  |s-s_exact|_2 = %.3f m,  |s-s_exact|_av = %.3f m" % (err_l2, err_av))

    # report glaciated area and inactive set agreement using Jaccard index
    ei = amr._eleminactive(u, lb)
    area = assemble(ei * dx)
    print(
        f"  glaciated area {area / 1000.0**2:.2e} km^2", end="" if i > 0 else "\n"
    )
    if i > 0:
        jac = amr.jaccard(ei, oldei, submesh=True)
        print(f"; levels {i-1},{i} Jaccard agreement {100*jac:.2f}%")
    oldei = ei

# save results from final mesh
if args.opvd:
    CU = ((n + 2) / (n + 1)) * Gamma
    U_ufl = CU * H ** p * inner(grad(s), grad(s)) ** ((p - 2) / 2) * grad(s)
    U = Function(VectorFunctionSpace(mesh, "CG", degree=2))
    U.project(secpera * U_ufl)  # smoother than .interpolate()
    U.rename("U = surface velocity (m/a)")
    q = Function(FunctionSpace(mesh, "BDM", 1))
    q.interpolate(U * H)
    q.rename("q = UH = *post-computed* ice flux")
    rank = Function(FunctionSpace(mesh, "DG", 0))
    rank.dat.data[:] = mesh.comm.rank
    rank.rename("rank")
    print("writing to %s ..." % args.opvd)
    if args.prob == "dome":
        VTKFile(args.opvd).write(a, u, s, H, U, q, sexact, sdiff, rank)
    else:
        Gb = Function(VectorFunctionSpace(mesh, "DG", degree=0))
        Gb.interpolate(grad(b))
        Gb.rename("Gb = grad(b)")
        VTKFile(args.opvd).write(a, u, s, H, U, q, b, Gb, rank)
