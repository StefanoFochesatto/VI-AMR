from argparse import ArgumentParser, RawTextHelpFormatter

parser = ArgumentParser(description="""
Solves a 2D steady, isothermal shallow ice approximation glacier obstacle problem.

Synthetic problem (default):  The domain is a square [0,L]^2 where L = 1800.0 km.
By default (-prob cap) we generate a random, but smooth, bed topography.  Option
-prob dome solves a flat bed case where the exact solution is known.  Option
-prob range generates a disconnected glacier.

Data-based problem (-data DATA.nc):
    FIXME: WIP

The solver is vinewtonrsls + mumps.  We applies AMR by the VCD method.  (Note UD0
is not currently parallel.)  The default refinement mode is to alternate VCD-only
on even refinements and VCD *with all of the inactive set marked* on odd refinements.

Examples:
  python3 steady.py -opvd cap.pvd
  python3 steady.py -prob dome -opvd dome.pvd
  mpiexec -n 4 python3 steady.py -irefine -refine 4 -prob range -opvd range4.pvd

This high resolution example ends with 2.2e6 elements and 250 m resolution along
glacier margin:
  mpiexec -n 12 python3 steady.py -prob range -m 200 -refine 5 -opvd range.pvd

Works in serial only:
  python3 steady.py -jaccard
""", formatter_class=RawTextHelpFormatter)
parser.add_argument('-data', metavar='FILE', type=str, default='',
                    help='read "topg" variable from NetCDF file (.nc)')
parser.add_argument('-jaccard', action='store_true', default=False,
                    help='use VIAMR.jaccard() to evaluate glaciated area convergence')
parser.add_argument('-m', type=int, default=20, metavar='M',
                    help='number of cells in each direction [default=20]')
parser.add_argument('-opvd', metavar='FILE', type=str, default='',
                    help='output file name for Paraview format (.pvd)')
parser.add_argument('-freezecount', type=int, default=5, metavar='P',
                    help='number of frozen-tilt iterations [default=5]')
parser.add_argument('-newton', action='store_true', default=False,
                    help='use straight Newton, which is less robust than default (=iterate on frozen "tilt")')
parser.add_argument('-prob', type=str, default='cap', metavar='X',
                    choices = ['cap','dome','range'],
                    help='choose problem from {cap, dome, range}')
parser.add_argument('-refine', type=int, default=2, metavar='R',
                    help='number of AMR refinements [default 2]')
parser.add_argument('-rmethod', type=str, default='alternate', metavar='X',
                    choices = ['alternate','vcdonly','always'],
                    help='choose refinement agenda from {alternate, vcdonly, always}')
parser.add_argument('-weakform', type=str, default='primal', metavar='X',
                    choices = ['primal','mixed'],
                    help='choose weak form from {primal, mixed}')
args, passthroughoptions = parser.parse_known_args()

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
assert args.refine >= 0, 'cannot refine a negative number of times'

# read data for bed topography into numpy array
if args.data:
    printpar('ignoring -prob choice ...')
    printpar(f'reading topg from NetCDF file {args.data} with native data grid:')
    topg_nc = DataNetCDF(args.data, 'topg')
    #topg_nc.preview()
    topg_nc.describe_grid(print=printpar, indent=4)
    printpar(f'putting topg onto Firedrake structured data mesh matching native grid ...')
    topg, nearb = topg_nc.function(delnear=100.0e3)
else:
    printpar(f'generating synthetic {args.m} x {args.m} initial mesh for problem {args.prob} ...')

if args.data:
    # generate netgen mesh compatible with data mesh, but unstructured
    # and at user (-m) resolution, typically lower
    mesh = topg_nc.ngmesh(args.m)
else:
    # generate via Firedrake
    mesh = RectangleMesh(args.m, args.m, L, L)

# solver parameters
sp = {"snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-3,  # roughly within 1 part in 10^-12 for u=H^{8/3}
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-8,
    "snes_stol": 1.0e-5,
    #"snes_monitor": None,
    #"snes_vi_monitor": None,
    "snes_converged_reason": None,
    #"snes_linesearch_type": "basic",
    "snes_linesearch_type": "bt",
    "snes_linesearch_order": "1",
    "snes_max_it": 1000,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"}

# transformed SIA
p = n + 1                # typical:  p = 4
omega = (p - 1) / (2*p)  #           omega = 3/8
phi = (p + 1) / (2*p)    #           phi = 5/8
r = p / (p - 1)          #           r = 4/3

def Phi(u, b):
    return - (1.0 / omega) * (u + 1.0)**phi * grad(b)  # FIXME the +1 regularization seems needed?

# FIXME failed ultraweak form ...
# def P(q):
#     gam = Gamma * omega**(p-1)
#     C = gam**(-1.0/(p-1))
#     return C * (inner(q, q) + 0.01)**((r - 2)/2)  # FIXME mixed regularization 0.01 too critical here
# def weakformMIXED(qu, a, b):
#    q, u = split(qu)
#    w, v = TestFunctions(qu.function_space())
#    return (div(q) - a) * v * dx - u * div(w) * dx + inner(P(q) * q - Phi(u, b), w) * dx

def weakform_mixed(qu, a, b, Z=None):
    q, u = split(qu)
    w, v = TestFunctions(qu.function_space())
    if Z is not None:
        du_tilt = grad(u) - Z
    else:
        du_tilt = grad(u) - Phi(u, b)
    Dp = inner(du_tilt, du_tilt)**((p-2)/2)
    gam = Gamma * omega**(p-1)
    return (div(q) - a) * v * dx(degree=2) \
           + inner(q + gam * Dp * du_tilt, w) * dx(degree=4)  # results non-sensical with quadrature degree 1 (though best for accuracy in dome test!?)
# OBSERVATION:  looking at solution q shows last term needs sufficient quadrature degree, possibly BDM degree plus 2?

def weakform_primal(u, a, b, Z=None):
    v = TestFunction(u.function_space())
    if Z is not None:
        du_tilt = grad(u) - Z
    else:
        du_tilt = grad(u) - Phi(u, b)
    Dp = inner(du_tilt, du_tilt)**((p-2)/2)
    return Gamma * omega**(p-1) * Dp * inner(du_tilt, grad(v)) * dx - a * v * dx

# outer mesh refinement loop
amr = VIAMR(debug=True)
for i in range(args.refine + 1):
    # mark and refine
    if i > 0:
        if args.jaccard:
            # generate active set indicator so we can evaluate Jaccard index
            eactive = amr.elemactive(s, lb)
        mark = amr.vcdmark(mesh, s, lb)  # good
        #mark = amr.udomark(mesh, s, lb, n=1)  # not in parallel, but otherwise good
        if ((args.rmethod == 'alternate' and i % 2 == 1) or args.rmethod == 'always'):
            imark = amr.eleminactive(s, lb)
            _, DG0 = amr.spaces(mesh)
            mark = Function(DG0).interpolate((mark + imark) - (mark * imark)) # union
        mesh = amr.refinemarkedelements(mesh, mark)

    if args.weakform == 'mixed':
        # spaces for flux q and transformed thickness u on current mesh
        W = FunctionSpace(mesh, "BDM", 1)
        V = FunctionSpace(mesh, "CG", 1)
        Z = W * V
    elif args.weakform == 'primal':
        # space for transformed thickness u on current mesh
        V = FunctionSpace(mesh, "CG", 1)
    else:
        raise NotImplementedError('unknown option to -weakform')

    # obstacle and source term
    if args.data:
        lb = Function(V).project(topg) # cross-mesh projection from data mesh
        # SMB from linear model based on lapse rate; from linearizing dome case
        c0 = -3.4e-8
        c1 = (6.3e-8 - c0) / 3.6e3
        a_lapse = c0 + c1 * topg
        a = Function(V).interpolate(conditional(nearb > 0.0, -1.0e-6, a_lapse)) # also cross-mesh re nearb
    else:
        x = SpatialCoordinate(mesh)
        if args.prob == 'dome':
            lb = Function(V).interpolate(Constant(0))
            sexact = Function(V).interpolate(dome_exact(x))
            sexact.rename("s_exact")
        else:
            lb = Function(V).interpolate(bumps(x, problem=args.prob))
        a = Function(V).interpolate(accumulation(x, problem=args.prob))
    lb.rename("lb = bedrock topography")
    a.rename("a = accumulation")

    # initialize transformed thickness variable
    if i == 0:
        # build pile of ice from accumulation
        pileage = 400.0  # years
        Hinit = pileage * secpera * conditional(a > 0.0, a, 0.0)
        uold = Function(V).interpolate(Hinit**omega)
    else:
        # cross-mesh interpolation of previous solution
        uold = Function(V).interpolate(u)
        # remove sign flaws from cross-mesh interpolation
        uold = Function(V).interpolate(conditional(uold < 0.0, 0.0, uold))

    if args.weakform == 'mixed':
        qu = Function(Z)
        qu.subfunctions[1].interpolate(uold)
    else:
        u = Function(V, name="u = transformed thickness").interpolate(uold)

    # set-up for solve on current mesh
    nv, ne, hmin, hmax = amr.meshsizes(mesh)
    printpar(f'solving problem {args.prob} using weak form {args.weakform} on mesh level {i}:')
    printpar(f'  {nv} vertices, {ne} elements, h in [{hmin/1e3:.3f},{hmax/1e3:.3f}] km ...')

    if args.weakform == 'mixed':
        ninf, inf = PETSc.NINFINITY, PETSc.INFINITY
        lower = Function(Z)
        lower.subfunctions[0].dat.data[:] = ninf
        lower.subfunctions[1].dat.data[:] = 0.0
        upper = Function(Z)
        upper.subfunctions[0].dat.data[:] = inf
        upper.subfunctions[1].dat.data[:] = inf
        bcs = [DirichletBC(Z.sub(0), as_vector([0.0, 0.0]), "on_boundary"),]
    else:
        lower = Function(V).interpolate(Constant(0.0))
        upper = Function(V).interpolate(Constant(PETSc.INFINITY))
        bcs = [DirichletBC(V, Constant(0.0), "on_boundary"),]

    if args.newton:
        if args.weakform == 'mixed':
            F = weakform_mixed(qu, a, lb)
            problem = NonlinearVariationalProblem(F, qu, bcs=bcs)
        else:
            F = weakform_primal(u, a, lb)
            problem = NonlinearVariationalProblem(F, u, bcs=bcs)
        solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="s")
        solver.solve(bounds=(lower, upper))
    else:
        # outer loop for freeze iteration
        for k in range(args.freezecount):
            printpar(f'  freeze tilt iteration {k+1} ...')
            Ztilt = Phi(uold, lb)
            if args.weakform == 'mixed':
                F = weakform_mixed(qu, a, lb, Z=Ztilt)
                problem = NonlinearVariationalProblem(F, qu, bcs=bcs)
            else:
                F = weakform_primal(u, a, lb, Z=Ztilt)
                problem = NonlinearVariationalProblem(F, u, bcs=bcs)
            solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="s")
            solver.solve(bounds=(lower, upper))
            if args.weakform == 'mixed':
                q, u = qu.subfunctions
            uold = Function(V).interpolate(u)

    # update true geometry variables
    if args.weakform == 'mixed':
        q, u = qu.subfunctions
        q.rename("q = ice flux")
        u.rename("u = transformed thickness")
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
        neweactive = amr.elemactive(s, lb)
        jac = amr.jaccard(eactive, neweactive)
        printpar(f'  glaciated areas Jaccard agreement {100*jac:.2f}% [levels {i-1}, {i}]' )
        eactive = neweactive

# on final, finest mesh, save results
if args.opvd:
    CU = ((n+2)/(n+1)) * Gamma
    U_ufl = CU * (s - lb)**p * inner(grad(s), grad(s))**((p-2)/2) * grad(s)
    U = Function(VectorFunctionSpace(mesh, 'CG', degree=2))
    U.project(secpera * U_ufl)  # smoother than .interpolate()
    U.rename("U = surface velocity (m/a)")
    if args.weakform != 'mixed':
        q = Function(FunctionSpace(mesh, 'BDM', 1))
        q.interpolate(U * (s - lb))
        q.rename("q = UH = *post-computed* ice flux")
    rank = Function(FunctionSpace(mesh, 'DG', 0))
    rank.dat.data[:] = mesh.comm.rank
    rank.rename('rank')
    printpar('writing to %s ...' % args.opvd)
    if args.prob == 'dome':
        VTKFile(args.opvd).write(a,u,s,H,U,q,sexact,sdiff,rank)
    else:
        Gb = Function(VectorFunctionSpace(mesh, 'DG', degree=0))
        Gb.interpolate(grad(lb))
        Gb.rename('Gb = grad(b)')
        VTKFile(args.opvd).write(a,u,s,H,U,q,lb,Gb,rank)
