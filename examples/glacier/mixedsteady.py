# TODO * allow other strategies than alternating AMR & uniform?

from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(description="""
Solves 2D steady, isothermal shallow ice approximation glacier obstacle problem.
Synthetic problem (default):
    Domain is a square [0,L]^2 where L = 1800.0 km. By default generates
    a random, but smooth, bed topography.  Option -prob dome solves in
    a flat bed case where the exact solution is known, and -prob range
    generates a disconnected glacier.
Data-based problem (-data DATA.nc):
    FIXME WIP
FE approximation is mixed/dual where transformed thickness variable u(x,y)
is DG0 and flux q(x,y) is BDM1.  See METHOD.md.  Solver is vinewtonrsls + mumps.
Applies VIAMR VCES method because UD0 is not currently parallel.
Refinement schedule is alternation with uniform refinement.
Examples:
  python3 steady.py -opvd cap.pvd
  python3 steady.py -prob dome -opvd dome.pvd
Works in serial only:
  python3 steady.py -jaccard
Also runs in parallel:
  mpiexec -n 4 steady.py -opvd cap4.pvd
FIXME DOCUMENT BETTER
""", formatter_class=RawTextHelpFormatter)
parser.add_argument('-data', metavar='FILE', type=str, default='',
                    help='read "topg" variable from NetCDF file (.nc)')
parser.add_argument('-jaccard', action='store_true', default=False,
                    help='use VIAMR.jaccard() to evaluate glaciated area convergence')
parser.add_argument('-m', type=int, default=20, metavar='M',
                    help='number of cells in each direction [default=20]')
parser.add_argument('-onelevel', action='store_true', default=False,
                    help='no AMR refinements; use Firedrake to generate uniform mesh')
parser.add_argument('-opvd', metavar='FILE', type=str, default='',
                    help='output file name for Paraview format (.pvd)')
parser.add_argument('-prob', type=str, default='cap', metavar='X',
                    choices = ['cap','dome','range'],
                    help='choose problem from {cap,dome,range}')
parser.add_argument('-refine', type=int, default=2, metavar='R',
                    help='number of AMR refinements [default 2]')
args, passthroughoptions = parser.parse_known_args()

# up to 12 levels of refinement according to schedule
rdict = {'u': 'uniform',  # each triangle becomes 4
         'v': 'VCES',     # apply vcesmark() with default parameters
         'd': 'UDO'}      # apply udomark() with n=1
rsched = [None, 'u', 'v', 'u', 'v', 'u', 'v', 'u', 'v', 'u', 'v', 'u', 'v']
# FIXME allow alternative schedules

import numpy as np
import petsc4py
petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
printpar = PETSc.Sys.Print
from viamr import VIAMR

from synthetic import secpera, n, Gamma, L, dome_exact, accumulation, bumps
from datanetcdf import DataNetCDF

assert args.m >= 1, 'at least one cell in mesh'
if args.onelevel:
    if args.data:
        raise NotImplementedError('incompatible arguments -onelevel -data')
    args.refine = 0
    printpar('not using refinement; uniform mesh generated with Firedrake')

# read data for bed topography into numpy array
if args.data:
    printpar('ignoring -prob choice ...')
    printpar(f'reading topg from NetCDF file {args.data} with native data grid:')
    topg_nc = DataNetCDF(args.data, 'topg')
    #topg_nc.preview()
    topg_nc.describe_grid(print=printpar, indent=4)
    printpar(f'putting topg onto Firedrake structured data mesh matching native grid ...')
    topg_native, nearb_native = topg_nc.function(delnear=100.0e3)
else:
    printpar(f'generating synthetic {args.m} x {args.m} initial mesh for problem {args.prob} ...')

if args.onelevel:
    # generate via Firedrake
    mesh = RectangleMesh(args.m, args.m, L, L)
else:
    if args.data:
        # generate netgen mesh compatible with data mesh, but unstructured
        # and at user (-m) resolution, typically lower
        mesh = topg_nc.ngmesh(args.m)
    else:
        # prepare for AMR by generating via netgen
        try:
            import netgen
        except ImportError:
            printpar("ImportError.  Unable to import NetGen.  Exiting.")
            import sys
            sys.exit(0)
        from netgen.geom2d import SplineGeometry
        geo = SplineGeometry()
        geo.AddRectangle(p1=(0.0, 0.0), p2=(L, L), bc="rectangle")
        trih = L / args.m
        ngmsh = geo.GenerateMesh(maxh=trih)
        mesh = Mesh(ngmsh)

# solver parameters
sp = {"snes_type": "vinewtonrsls",
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-8,
    "snes_stol": 1.0e-5,
    #"snes_monitor": None,
    "snes_converged_reason": None,
    "snes_linesearch_type": "bt",
    "snes_linesearch_order": "1",
    "snes_max_it": 1000,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"}

# transformed SIA
p = n + 1
r = p / (p - 1)
omega = (p - 1) / (2*p)
phi = (p + 1) / (2*p)

def Phi(u, b):
    return - (1.0 / omega) * (u + 1.0)**phi * grad(b)  # FIXME regularization

def B(Z, uq, a, b):
    u, q = split(uq)
    v, w = TestFunctions(Z)
    lilgamma = Gamma * omega**(p - 1)
    C = lilgamma**(-1.0 / (p - 1))
    qpow = (1.0 + inner(q, q))**((r - 2) / 2)  # FIXME regularization
    return - u * div(w) * dx \
           + inner(C * qpow * q - Phi(u, b), w) * dx \
           + (div(q) - a) * v * dx


# outer mesh refinement loop
for i in range(args.refine + 1):
    if i > 0 and args.jaccard:
        # generate active set indicator so we can evaluate Jaccard index
        eactive = VIAMR(debug=True).elemactive(s, lb)

    # mark and refine according to schedule
    if rsched[i] is not None:
        if rsched[i] == 'u':
            W = FunctionSpace(mesh, "DG", 0)
            mark = Function(W).interpolate(Constant(1.0))  # mark everybody
        elif rsched[i] == 'v':
            mark = VIAMR().vcesmark(mesh, s, lb)  # good
        elif rsched[i] == 'd':
            mark = VIAMR().udomark(mesh, s, lb, n=1)  # not in parallel, but otherwise good
        else:
            raise NotImplementedError
        mesh = mesh.refine_marked_elements(mark)  # NetGen gives next mesh

    # mixed function space on current mesh
    V = FunctionSpace(mesh, "DG", 0)
    W = FunctionSpace(mesh, "BDM", 1)  # vector-valued, but this is correct
    Z = V * W

    # obstacle and source term
    CG1 = FunctionSpace(mesh, "CG", 1)
    if args.data:
        topg = Function(CG1).project(topg_native) # cross-mesh projection from data mesh
        # SMB from linear model based on lapse rate; from linearizing dome case
        c0 = -3.4e-8
        c1 = (6.3e-8 - c0) / 3.6e3
        a_lapse = c0 + c1 * topg_native
        a = Function(CG1).interpolate(conditional(nearb_native > 0.0, -1.0e-6, a_lapse)) # also cross-mesh
    else:
        x = SpatialCoordinate(mesh)
        if args.prob == 'dome':
            topg = Function(CG1).interpolate(Constant(0.0))
            sexact = Function(V).interpolate(dome_exact(x))
            sexact.rename("s_exact")
        else:
            topg = Function(CG1).interpolate(bumps(x, problem=args.prob))
        a = Function(CG1).interpolate(accumulation(x, problem=args.prob))
    topg.rename("topg = bedrock topography")
    a.rename("a = accumulation")

    # initialize transformed thickness variable
    uq = Function(Z)
    u, q = uq.subfunctions
    if i == 0:
        # build pile of ice from accumulation
        pileage = 400.0  # years
        Hinit = pileage * secpera * conditional(a > 0.0, a, 0.0)
        uold = Function(V).interpolate(Hinit**omega)
    else:
        # cross-mesh interpolation of previous solution
        uold = Function(V).interpolate(u)
    # remove sign flaws from cross-mesh interpolation (FIXME still needed for DG0?)
    uold = Function(V).interpolate(conditional(uold < 0.0, 0.0, uold))
    u.interpolate(uold)
    q.interpolate(as_vector([Constant(0.0),Constant(0.0)]))

    # DEBUG look at the data
    u.rename('u = transformed thickness')
    q.rename('q = ice flux')
    VTKFile('data.pvd').write(u, q, a, topg)

    # describe current mesh
    nv, ne, hmin, hmax = VIAMR().meshsizes(mesh)
    rstr = '' if rsched[i] is None else f' (after refine by {rdict[rsched[i]]})'
    printpar(f'solving level {i}{rstr}: {nv} vertices, {ne} elements, h in [{hmin/1e3:.3f},{hmax/1e3:.3f}] km ...')

    # boundary values and bounds
    bcs = DirichletBC(Z.sub(0), Constant(0.0), "on_boundary")
    inf, ninf = Constant(PETSc.INFINITY), Constant(PETSc.NINFINITY)
    lower = Function(Z)
    lower.subfunctions[0].interpolate(Constant(0.0))
    lower.subfunctions[1].interpolate(as_vector([ninf, ninf]))
    upper = Function(Z)
    upper.subfunctions[0].interpolate(inf)
    upper.subfunctions[1].interpolate(as_vector([inf, inf]))

    # weakform and solve
    F = B(Z, uq, a, topg)
    problem = NonlinearVariationalProblem(F, uq, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="")
    solver.solve(bounds=(lower, upper))
    u, q = uq.subfunctions
    u.rename('u = transformed thickness')
    q.rename('q = ice flux')

    # update true geometry variables
    H = Function(V, name="H = thickness").interpolate(u**omega)
    s = Function(V, name="s = surface elevation").interpolate(lb + H)

    # report numerical errors if exact solution known
    if args.prob == 'dome':
        sdiff = Function(V).interpolate(s - dome_exact(x))
        sdiff.rename("sdiff = s - s_exact")
        err_l2 = norm(sdiff / L)
        err_av = norm(sdiff, 'l1') / L**2
        printpar('    |s-s_exact|_2 = %.3f m,  |s-s_exact|_av = %.3f m' \
                 % (err_l2, err_av))

    # report active set agreement between consecutive meshes using Jaccard index
    if i > 0 and args.jaccard:
        z = VIAMR(debug=True)
        neweactive = z.elemactive(s, lb)
        jac = z.jaccard(eactive, neweactive)
        printpar(f'  glaciated areas Jaccard agreement {100*jac:.2f}% [levels {i-1}, {i}]' )
        eactive = neweactive

# on final, finest mesh, save results
if args.opvd:
    printpar('writing to %s ...' % args.opvd)
    if args.prob == 'dome':
        VTKFile(args.opvd).write(a,u,q,s,H,sexact,sdiff)
    else:
        Gb = Function(VectorFunctionSpace(mesh, 'DG', degree=0))
        Gb.interpolate(grad(topg))
        Gb.rename('Gb = grad(b)')
        VTKFile(args.opvd).write(a,u,q,s,H,topg,Gb)
