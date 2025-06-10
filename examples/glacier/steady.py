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

An elevation-dependent surface mass balance model is also available, with
options -elevdepend (to turn on) and -sELA (equilibrium line altitude).
This case does not allow -newton.

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
  mpiexec -n 12 python3 steady.py -m 5 -refine 13 -newton   # 31 m margin resolution

The -newton solve works well but needs a coarser initial grid:
  mpiexec -n 20 python3 steady.py -prob range -newton -m 5 -refine 10 -rmethod alternate -opvd result_rangenewton.pvd

Elevation-dependent surface mass balance example:
  mpiexec -n 4 python3 steady.py -prob cap -elevdepend -sELA 1000.0 -m 20 -uniform 1 -udo_n 2 -theta 0.2 -pcount 20 -refine 6 -opvd result_cap_ed.pvd
Vary -sELA 900.0|800.0|700.0 to see rapid increase in inactive set area (glaciation).

FIXME: OLD -data run:
  python3 steady.py -data eastgr.nc -opvd result_data.pvd -refine 5 -pcount 24
""",
    formatter_class=RawTextHelpFormatter,
)
parser.add_argument(
    "-csv",
    metavar="FILE",
    type=str,
    default="",
    help="output file name for dome error report (.csv)",
)
parser.add_argument(
    "-data",
    metavar="FILE",
    type=str,
    default="",
    help='read "topg" variable from NetCDF file (.nc)',
)
parser.add_argument(
    "-elevdepend",
    action="store_true",
    default=False,
    help="compute surface mass balance from an elevation-dependent model",
)
parser.add_argument(
    "-jaccard",
    action="store_true",
    default=False,
    help="compare successive active sets by Jaccard agreement",
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
    help="use straight Newton instead of Picard+Newton",
)
parser.add_argument(
    "-opvd",
    metavar="FILE",
    type=str,
    default="",
    help="output file name for Paraview format (.pvd)",
)
parser.add_argument(
    "-pcount",
    type=int,
    default=10,
    metavar="P",
    help="number of Picard frozen-tilt (and a(s) if -elevdepend) iterations [default=10]",
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
    "-sELA",
    type=float,
    default=1000.0,
    metavar="X",
    help="equilibrium line altitude to use if -elevdepend [default=1000.0]",
)
parser.add_argument(
    "-softening",
    type=float,
    default=1.0,
    metavar="X",
    help="multiply Gamma by softening factor X; X>1 softens, 0<X<1 hardens [default=1.0]",
)
parser.add_argument(
    "-theta",
    type=float,
    default=0.5,
    metavar="X",
    help="theta to use in fixed-rate marking strategy in inactive set [default=0.5]",
)
parser.add_argument(
    "-udo_n",
    type=int,
    default=1,
    metavar="N",
    help="use udomark(.., n=N) [default 1]",
)
parser.add_argument(
    "-uniform",
    type=int,
    default=0,
    metavar="R",
    help="initial R refinements are uniform [default 0]",
)
parser.add_argument(
    "-vcd",
    action="store_true",
    default=False,
    help="apply VCD free-boundary marking (instead of UDO)",
)
args, passthroughoptions = parser.parse_known_args()

assert args.m >= 1, "at least one cell in mesh"
assert args.refine >= 0, "cannot refine a negative number of times"
assert args.pcount >= 1, "at least one Picard iteration required"
assert args.udo_n >= 0, "cannot use UDO with negative levels"
assert not args.elevdepend or args.prob != "dome", "combination invalid: -elevdepend & -prob dome"
assert not args.elevdepend or not args.data, "combination invalid: -elevdepend & -data file.nc"
assert not args.elevdepend or not args.newton, "combination invalid: -elevdepend & -newton"  # FIXME

import numpy as np
import petsc4py

petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI

pprint = PETSc.Sys.Print  # parallel print
from viamr import VIAMR

from synthetic import secpera, n, Gamma, L, dome_exact, accumulation, bumps, domeL, domeH0

# set up .csv if generating numerical error data
if args.csv:
    if not args.prob == "dome":
        raise ValueError("option -csv only valid for -prob dome")
    csvfile = open(args.csv, 'w')
    print("REFINE,NE,HMIN,UERRH1,HERRINF,DRMAX", file=csvfile)

# read data for bed topography
if args.data:
    pprint("ignoring -prob choice ...")
    args.prob = None
    pprint(f"reading topg from NetCDF file {args.data} with native data grid:")
    from datanetcdf import DataNetCDF

    topg_nc = DataNetCDF(args.data, "topg")
    #topg_nc.preview()
    topg_nc.describe_grid(print=PETSc.Sys.Print, indent=4)
    pprint(f"putting topg onto matching Firedrake structured data mesh ...")
    topg, nearb = topg_nc.function(delnear=100.0e3)
else:
    pprint(
        f"generating synthetic {args.m} x {args.m} initial mesh for problem {args.prob} ..."
    )

# setting distribution parameters should not be necessary ...
dp = {
    "partition": True,
    "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
}
if args.data:
    # generate mesh compatible with data mesh, but at user (-m) resolution, typically lower
    mesh = topg_nc.rectmesh(args.m)
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
    #"snes_max_funcs": 10000,
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


def amodel(s, sELA=1000.0, dsNEXT=100.0, alpha=0.0001 / secpera, alpharat=0.01):
    """Model of surface mass balance a(s) where alpha is lapse rate below sELA
    and above sELA there is a lower-slope (by alpharat) logarithmic function."""
    tau = dsNEXT - sELA
    beta = alpharat * alpha * dsNEXT
    return conditional(s < sELA, alpha * (s - sELA), beta * (ln(s + tau) - ln(dsNEXT)))


def weakform(u, a, b, Z=None, softening=1.0):
    v = TestFunction(u.function_space())
    if Z is not None:
        du_tilt = grad(u) - Z
    else:
        du_tilt = grad(u) - Phi(u, b)
    Dp = inner(du_tilt, du_tilt) ** ((p - 2) / 2)
    return softening * Gamma * omega ** (p - 1) * Dp * inner(du_tilt, grad(v)) * dx - a * v * dx


def glaciermeshreport(amr, mesh, indent=2):
    nv, ne, hmin, hmax = amr.meshsizes(mesh)
    hmin /= 1000.0
    hmax /= 1000.0
    indentstr = indent * " "
    PETSc.Sys.Print(
        f"{indentstr}current mesh: {nv} vertices, {ne} elements, h in [{hmin:.3f},{hmax:.3f}] km"
    )
    return None


def normerrorsdome(uh, Hh):
    """Return relative H^1 norm error in u and L^infty norm error in H.
    For the first, generate uexact in better space (and UFL from
    dome_exact()).  For L^infty error in H we merely want the max nodal
    error, so using V=CG1 is fine."""
    V = uh.function_space()
    x = SpatialCoordinate(V.mesh())
    Hdiff = Function(V).interpolate(Hh - dome_exact(x))
    Hdiff.rename("Hdiff = H - Hexact")
    with Hdiff.dat.vec_ro as v:
        Herr = abs(v).max()[1]
    CG2 = FunctionSpace(V.mesh(), "CG", 2)
    uexact = Function(CG2).interpolate(dome_exact(x)**(1.0/omega))
    uexact.rename("uexact")
    uerr = errornorm(uexact, u, norm_type="H1") / norm(uexact, norm_type="H1")
    return uerr, Herr


def radiuserrordome(amr, uh):
    """For -prob "dome", compute the maximum of free-boundary radius
    errors from a solution uh.  The exact free boundary is a circle
    of radius domeL with center (L/2,L/2).  Returns the maximum
    radius error."""
    V = uh.function_space()
    vfb, _ = amr.freeboundarygraph(uh, Function(V).interpolate(0.0))
    vfb = np.array(vfb)
    mymax = PETSc.NINFINITY
    if len(vfb) > 0:
        x, y = vfb[:,0], vfb[:,1]
        drfb = np.abs(np.sqrt((x - L/2)**2 + (y - L/2)**2) - domeL)
        mymax = np.max(drfb)
    drmax = float(V.mesh().comm.allreduce(mymax, op=MPI.MAX))
    return drmax


# outer mesh refinement loop
amr = VIAMR(debug=True)
for i in range(args.refine + 1):
    # mark and refine based on constraint u >= 0
    if i > 0:
        if i < args.uniform + 1:
            pprint(f"refining uniformly ...")
            _, DG0 = amr.spaces(mesh)
            mark = Function(DG0).interpolate(Constant(1.0))
            mesh = amr.refinemarkedelements(mesh, mark, isUniform=True)
        else:
            pprint(f"refining free boundary by {'VCD' if args.vcd else 'UDO'}", end="")
            if args.vcd:
                # change bracket vs default [0.2, 0.8], to provide more high-res
                #   for ice near margin (0.2 -> 0.1), i.e. on inactive side
                fbmark = amr.vcdmark(u, lb, bracket=[0.1, 0.8])
            else:
                fbmark = amr.udomark(u, lb, n=args.udo_n)
            pprint(", and by gradient recovery in inactive ...")
            # FIXME: sporadic parallel bug with method="total" apparently ...
            #imark, _, _ = amr.gradrecinactivemark(u, lb, theta=args.theta, method="total")
            imark, _, _ = amr.gradrecinactivemark(u, lb, theta=args.theta, method="max")
            mark = amr.unionmarks(fbmark, imark)
            mesh = amr.refinemarkedelements(mesh, mark)
            # report percentages of elements marked
            inactive = amr._eleminactive(u, lb)
            perfb = 100.0 * amr.countmark(fbmark) / ne
            perin = 100.0 * amr.countmark(imark) / amr.countmark(inactive)
            pprint(f"  {perfb:.2f}% all elements free-boundary marked, {perin:.2f}% inactive elements marked")

    # describe current mesh
    nv, ne, hmin, hmax = amr.meshsizes(mesh)
    pprint(f"solving problem {args.prob} on mesh level {i}:")
    glaciermeshreport(amr, mesh)

    # space for most functions
    V = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)

    # bedrock on current mesh
    if args.data:
        b = Function(V).project(topg)  # cross-mesh projection from data mesh
    else:
        if args.prob == "dome":
            b = Function(V).interpolate(Constant(0.0))
        else:
            b = Function(V).interpolate(bumps(x, problem=args.prob))
    b.rename("b = bedrock topography")

    # surface mass balance function on current mesh; depends on b in one case
    if args.data:
        # SMB from linear model based on lapse rate; from linearizing dome case
        c0 = -3.4e-8
        c1 = (6.3e-8 - c0) / 3.6e3
        a_lapse = c0 + c1 * topg
        a = Function(V).interpolate(
            conditional(nearb > 0.0, -1.0e-6, a_lapse)
        )  # also cross-mesh re nearb
    elif args.elevdepend:
        # initialize from s = b assumption
        a = Function(V).interpolate(amodel(b, sELA=args.sELA))
    else:
        a = Function(V).interpolate(accumulation(x, problem=args.prob))
    a.rename("a = accumulation")

    # initialize transformed thickness variable; depends on b and a
    if i == 0:
        # build pile of ice from accumulation
        pileage = 400.0  # years
        Hinit = pileage * secpera * conditional(a > 0.0, a, 0.0)
        uold = Function(V).interpolate(Hinit ** (1.0 / omega))
    else:
        # cross-mesh interpolation of previous solution
        uold = Function(V).interpolate(u)
        # remove sign flaws from cross-mesh interpolation
        #   note: u = H^(8/3) < 1 is *very* little ice in an initial iterate
        uold = Function(V).interpolate(conditional(uold < 1.0, 0.0, uold))
    assert assemble(uold * dx) > 0, "initialization failure; u must correspond to positive ice"

    # solve on current mesh
    u = Function(V, name="u = transformed thickness").interpolate(uold)
    lb = Function(V).interpolate(Constant(0.0))  # lower bound *in solver*
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    bcs = [
        DirichletBC(V, Constant(0.0), "on_boundary"),
    ]
    if args.newton:
        F = weakform(u, a, b, softening=args.softening)
        problem = NonlinearVariationalProblem(F, u, bcs=bcs)
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="s"
        )
        solver.solve(bounds=(lb, ub))
    else:
        # outer loop for Picard (freeze-tilt) iteration, and a(s) if -elevdepend
        for k in range(args.pcount):
            # pprint(f'  Picard iteration {k+1} ...')
            if args.elevdepend:
                sold = b + uold ** omega
                a = Function(V).interpolate(amodel(sold, sELA=args.sELA))
                a.rename("a = accumulation")
            Ztilt = Phi(uold, b)
            F = weakform(u, a, b, Z=Ztilt, softening=args.softening)
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
    if not args.data and args.prob == "dome":
        uerr_H1, Herr_inf = normerrorsdome(u, H)
        drmax = radiuserrordome(amr, u)
        pprint(f"  |u-uexact|_H1 = {uerr_H1:.3e} rel, |H-Hexact|_inf = {Herr_inf:.3f} m, |dr|_inf = {drmax/1000.0:.3f} km")
        if args.csv:
            print(f"{i:d},{ne:d},{hmin:.2f},{uerr_H1:.3e},{Herr_inf:.3f},{drmax:.3f}", file=csvfile)

    # report glaciated area and inactive set agreement using Jaccard index
    vol = assemble(H * dx)
    ei = amr._eleminactive(u, lb)
    area = assemble(ei * dx)
    pprint(f"  glaciated area {area / 1000.0**4:.4f} million km^2, ice volume = {vol / 1000.0**4:.2f} thousand km^3", end="")
    if args.jaccard and i > 0:
        jac = amr.jaccard(ei, oldei, submesh=True)
        pprint(f"; levels {i-1},{i} Jaccard agreement {100*jac:.2f}%")
    else:
        pprint("")
    oldei = ei

if args.csv:
    csvfile.close()

# save results from final mesh
if args.opvd:
    CU = ((n + 2) / (n + 1)) * Gamma
    Us_ufl = CU * H ** p * inner(grad(s), grad(s)) ** ((p - 2) / 2) * grad(s)
    Us = Function(VectorFunctionSpace(mesh, "CG", degree=2))
    Us.project(secpera * Us_ufl)  # smoother than .interpolate()
    Us.rename("Us = surface velocity (m/a)")
    q = Function(FunctionSpace(mesh, "BDM", 1))
    q.interpolate(Us * H)
    q.rename("q = UH = *post-computed* ice flux")
    Gs = Function(VectorFunctionSpace(mesh, "DG", degree=0))
    Gs.interpolate(grad(s))
    Gs.rename("Gs = grad(s)")
    rank = Function(FunctionSpace(mesh, "DG", 0))
    rank.dat.data[:] = mesh.comm.rank
    rank.rename("rank")
    pprint("writing to %s ..." % args.opvd)
    if args.prob == "dome":
        VTKFile(args.opvd).write(u, H, Us, q, a, Gs, rank)
    else:
        Gb = Function(VectorFunctionSpace(mesh, "DG", degree=0))
        Gb.interpolate(grad(b))
        Gb.rename("Gb = grad(b)")
        VTKFile(args.opvd).write(u, H, s, Us, q, a, b, Gb, Gs, rank)
