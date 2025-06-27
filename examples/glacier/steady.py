# Solves a 2D steady, isothermal shallow ice approximation glacier obstacle problem.
# For more info see README.md or run
#   python3 steady.py -h

from clargs import parser

args, passthroughoptions = parser.parse_known_args()
assert args.m >= 1, "at least one cell in mesh"
assert args.refine >= 0, "cannot refine a negative number of times"
assert args.pcount >= 1, "at least one Picard iteration required"
assert args.udo_n >= 0, "cannot use UDO with negative levels"
assert (
    not args.elevdepend or args.prob != "dome"
), "combination invalid: -elevdepend & -prob dome"
assert (
    not args.elevdepend or not args.data
), "combination invalid: -elevdepend & -data file.nc"
assert (
    not args.elevdepend or not args.newton
), "combination invalid: -elevdepend & -newton"  # FIXME

import numpy as np
import petsc4py

petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI

pprint = PETSc.Sys.Print  # parallel print
from viamr import VIAMR

from synthetic import (
    secpera,
    n,
    Gamma,
    L,
    dome_exact,
    accumulation,
    bumps,
    domeL,
    domeH0,
    normerrorsdome,
    radiuserrordome,
)

if args.extractpvd:
    bx = args.box  # [x_left, x_right, y_lower, y_upper]
    assert bx[0] < bx[1] and bx[2] < bx[3], "-extract_box not valid"
    assert args.data or (0.0 <= bx[0] < bx[1] <= L), "x range not valid for [0,L]"
    assert args.data or (0.0 <= bx[2] < bx[3] <= L), "y range not valid for [0,L]"

# set up .csv if generating numerical error data
if args.csv:
    if not args.prob == "dome":
        raise ValueError("option -csv only valid for -prob dome")
    csvfile = open(args.csv, "w")
    print("REFINE,NE,HMIN,UERRH1,HERRINF,DRMAX", file=csvfile)

# read data for bed topography
if args.data:
    pprint("ignoring -prob choice ...")
    args.prob = None
    pprint(f"reading topg from NetCDF file {args.data} with native data grid:")
    from datanetcdf import DataNetCDF

    topg_nc = DataNetCDF(args.data, "topg")
    # topg_nc.preview()
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
    # "snes_max_funcs": 10000,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# transformed SIA
p = n + 1  # typical:  p = 4
omega = (p - 1) / (2 * p)  #  omega = 3/8
phi = (p + 1) / (2 * p)  #  phi = 5/8
r = p / (p - 1)  #  r = 4/3


def Beta(u, b):
    return (1.0 / omega) * (u + 1.0) ** phi * grad(b)  # eps=1 regularization is small


def amodel(s, sELA=1000.0, dsNEXT=100.0, alpha=0.0001 / secpera, alpharat=0.01):
    """Model of surface mass balance a(s) where alpha is lapse rate below sELA
    and above sELA there is a lower-slope (by alpharat) logarithmic function."""
    tau = dsNEXT - sELA
    beta = alpharat * alpha * dsNEXT
    return conditional(s < sELA, alpha * (s - sELA), beta * (ln(s + tau) - ln(dsNEXT)))


def weakform(u, a, b, Z=None, softening=1.0):
    """When Z=None this is the weak form corresponding to (3) in METHOD.md.
    If Z is given then this is (4).  In either case a(x) is given, so elevation
    -dependent surface mass balance *must* be handled by an outer iteration.
    Even for steep beds, the quadrature degree in the weak form, for the first
    "dx", can apparently be handled by Firedrake's automatic mechanism.  For
    testing this, note that "dx(degree=Q)" with Q=4,5,6,7 seems to produce about
    the same result as the automatic mechanism, while Q=2 is distinctly worse."""
    v = TestFunction(u.function_space())
    if Z is not None:
        du_tilt = grad(u) + Z
    else:
        du_tilt = grad(u) + Beta(u, b)
    Dp = inner(du_tilt, du_tilt) ** ((p - 2) / 2)
    C = softening * Gamma * omega ** (p - 1)
    return C * Dp * inner(du_tilt, grad(v)) * dx(degree=args.qdegree) - a * v * dx


def glaciermeshreport(amr, mesh, indent=2):
    nv, ne, hmin, hmax = amr.meshsizes(mesh)
    hmin /= 1000.0
    hmax /= 1000.0
    indentstr = indent * " "
    PETSc.Sys.Print(
        f"{indentstr}current mesh: {nv} vertices, {ne} elements, h in [{hmin:.3f},{hmax:.3f}] km"
    )
    return None


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
            # imark, _, _ = amr.gradrecinactivemark(u, lb, theta=args.theta, method="total")
            imark, _, _ = amr.gradrecinactivemark(u, lb, theta=args.theta, method="max")
            if args.hmin > 0.0:
                fbmark = amr.lowerboundcelldiameter(fbmark, args.hmin)
                imark = amr.lowerboundcelldiameter(imark, args.hmin)
            mark = amr.unionmarks(fbmark, imark)
            mesh = amr.refinemarkedelements(mesh, mark)
            # report percentages of elements marked
            inactive = amr._eleminactive(u, lb)
            perfb = 100.0 * amr.countmark(fbmark) / ne
            perin = 100.0 * amr.countmark(imark) / amr.countmark(inactive)
            pprint(
                f"  {perfb:.2f}% all elements free-boundary marked, {perin:.2f}% inactive elements marked"
            )

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
    assert (
        assemble(uold * dx) > 0
    ), "initialization failure; u must correspond to positive ice"

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
                sold = b + uold**omega
                a = Function(V).interpolate(amodel(sold, sELA=args.sELA))
                a.rename("a = accumulation")
            Ztilt = Beta(uold, b)
            F = weakform(u, a, b, Z=Ztilt, softening=args.softening)
            problem = NonlinearVariationalProblem(F, u, bcs=bcs)
            solver = NonlinearVariationalSolver(
                problem, solver_parameters=sp, options_prefix="s"
            )
            solver.solve(bounds=(lb, ub))
            uold = Function(V).interpolate(u)

    # update true geometry variables
    H = Function(V, name="H = thickness").interpolate(u**omega)
    s = Function(V, name="s = surface elevation").interpolate(b + H)

    # report numerical errors if exact solution known
    if not args.data and args.prob == "dome":
        uerr_H1, Herr_inf = normerrorsdome(u, H)
        vfb, _ = amr.freeboundarygraph(u, Function(V).interpolate(0.0))
        drmax = radiuserrordome(mesh, vfb)
        pprint(
            f"  |u-uexact|_H1 = {uerr_H1:.3e} rel, |H-Hexact|_inf = {Herr_inf:.3f} m, |dr|_inf = {drmax/1000.0:.3f} km"
        )
        if args.csv:
            print(
                f"{i:d},{ne:d},{hmin:.2f},{uerr_H1:.3e},{Herr_inf:.3f},{drmax:.3f}",
                file=csvfile,
            )

    # report glaciated area and inactive set agreement using Jaccard index
    vol = assemble(H * dx)
    ei = amr._eleminactive(u, lb)
    area = assemble(ei * dx)
    pprint(
        f"  glaciated area {area / 1000.0**4:.4f} million km^2, ice volume = {vol / 1000.0**4:.2f} thousand km^3",
        end="",
    )
    if args.jaccard and i > 0:
        jac = amr.jaccard(ei, oldei, submesh=True)
        pprint(f"; levels {i-1},{i} Jaccard agreement {100*jac:.2f}%")
    else:
        pprint("")
    oldei = ei

if args.csv:
    csvfile.close()

if args.extractpvd:  # note boxind gets written into -opvd file
    x, y = SpatialCoordinate(mesh)
    bx = args.box  # [x_left, x_right, y_lower, y_upper]
    ibx = conditional(x >= bx[0], conditional(x <= bx[1], 1.0, 0.0), 0.0)
    iby = conditional(y >= bx[2], conditional(y <= bx[3], 1.0, 0.0), 0.0)
    _, DG0 = amr.spaces(mesh)
    boxind = Function(DG0, name="box indicator").interpolate(ibx * iby)

if args.opvd:
    CU = ((n + 2) / (n + 1)) * Gamma
    Us_ufl = CU * H**p * inner(grad(s), grad(s)) ** ((p - 2) / 2) * grad(s)
    Us = Function(VectorFunctionSpace(mesh, "CG", degree=2))
    Us.project(secpera * Us_ufl)  # smoother than .interpolate()
    Us.rename("Us = surface velocity (m/a)")
    q = Function(FunctionSpace(mesh, "BDM", 1))
    q.interpolate(Us * H)
    q.rename("q = UH = *post-computed* ice flux")
    Gb = Function(VectorFunctionSpace(mesh, "DG", degree=0))
    Gb.interpolate(grad(b))
    Gb.rename("Gb = grad(b)")
    Gs = Function(VectorFunctionSpace(mesh, "DG", degree=0))
    Gs.interpolate(grad(s))
    Gs.rename("Gs = grad(s)")
    rank = Function(FunctionSpace(mesh, "DG", 0))
    rank.dat.data[:] = mesh.comm.rank
    rank.rename("rank")
    pprint("writing to %s ..." % args.opvd)
    if args.extractpvd:
        VTKFile(args.opvd).write(u, H, s, Us, q, a, b, Gb, Gs, rank, boxind)
    else:
        VTKFile(args.opvd).write(u, H, s, Us, q, a, b, Gb, Gs, rank)

if args.extractpvd:
    mesh.mark_entities(boxind, 99)
    mesh = RelabeledMesh(mesh, [boxind], [99])
    subm = Submesh(mesh, mesh.topological_dimension(), 99)
    subV = FunctionSpace(subm, "CG", 1)
    subu = Function(subV, name="u = transformed thickness").interpolate(u)
    subH = Function(subV, name="H = thickness").interpolate(H)
    subs = Function(subV, name="s = surface elevation").interpolate(s)
    suba = Function(subV, name="a = accumulation").interpolate(a)
    subb = Function(subV, name="b = bedrock topography").interpolate(b)
    pprint("writing box extract to %s ..." % args.extractpvd)
    VTKFile(args.extractpvd).write(subu, subH, subs, suba, subb)
