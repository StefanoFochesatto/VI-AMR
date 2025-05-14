from argparse import ArgumentParser, RawTextHelpFormatter

parser = ArgumentParser(description="""
Solves a 2D steady, isothermal shallow ice approximation glacier obstacle problem.

The default set-up is a synthetic problem over a square domain [0,L]^2 with
L = 1800.0 km.  A data-based problem (-data DATA.nc) is a WIP.

By default (-prob dome) we solve a flat bed case where the exact solution is
known.  Option -prob cap generates a random, but smooth, bed topography, but
keeps the dome SMB.  Option -prob range generates a different SMB and a
disconnected glacier.

We apply the VCD method for free-boundary refinement.  The default refinement
mode is to do VCD and gradient-recovery refinement, in the inactive set, on
every refinement.  The default VI solver is Picard iteration on the tilt
(Jouvet & Bueler, 2012), and vinewtonrsls + mumps for each tilt.

Examples:
  python3 steady.py -opvd dome.pvd
  mpiexec -n 4 python3 steady.py -refine 4 -prob range -opvd range.pvd

High-resolution example; achieved 30 m resolution along ice sheet margin:
  mpiexec -n 20 python3 steady.py -prob range -m 50 -refine 10 -opvd result_range.pvd
(Large memory needed ...)  Note L=1800 km so h_min / L ~ 1e-5.  However, such runs reveal less than perfect refinement right along the free boundary at very high resolution.
""", formatter_class=RawTextHelpFormatter)
parser.add_argument('-data', metavar='FILE', type=str, default='',
                    help='read "topg" variable from NetCDF file (.nc)')
parser.add_argument('-freezecount', type=int, default=8, metavar='P',
                    help='number of Picard frozen-tilt iterations [default=8]')
parser.add_argument('-jaccard', action='store_true', default=False,
                    help='use jaccard() to evaluate glaciated area convergence')
parser.add_argument('-m', type=int, default=20, metavar='M',
                    help='number of cells in each direction on initial mesh [default=20]')
parser.add_argument('-newton', action='store_true', default=False,
                    help='use straight Newton instead of Picard+Newton; not robust')
parser.add_argument('-opvd', metavar='FILE', type=str, default='',
                    help='output file name for Paraview format (.pvd)')
parser.add_argument('-prob', type=str, default='dome', metavar='X',
                    choices = ['dome','cap','range'],
                    help='choose problem from {dome, cap, range}')
parser.add_argument('-refine', type=int, default=2, metavar='R',
                    help='number of AMR refinements [default 2]')
parser.add_argument('-rmethod', type=str, default='always', metavar='X',
                    choices = ['always','alternate','vcdonly'],
                    help='choose refinement agenda from {always, alternate, vcdonly}')
args, passthroughoptions = parser.parse_known_args()

import numpy as np
import petsc4py
petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.petsc import PETSc
print = PETSc.Sys.Print
from viamr import VIAMR

from synthetic import secpera, n, Gamma, L, dome_exact, accumulation, bumps

assert args.m >= 1, 'at least one cell in mesh'
assert args.refine >= 0, 'cannot refine a negative number of times'

# read data for bed topography into numpy array
if args.data:
    print('ignoring -prob choice ...')
    print(f'reading topg from NetCDF file {args.data} with native data grid:')
    from datanetcdf import DataNetCDF
    topg_nc = DataNetCDF(args.data, 'topg')
    topg_nc.preview()
    topg_nc.describe_grid(print=print, indent=4)
    print(f'putting topg onto Firedrake structured data mesh matching native grid ...')
    topg, nearb = topg_nc.function(delnear=100.0e3)
else:
    print(f'generating synthetic {args.m} x {args.m} initial mesh for problem {args.prob} ...')

if args.data:
    # generate netgen mesh compatible with data mesh, but unstructured
    # and at user (-m) resolution, typically lower
    mesh = topg_nc.ngmesh(args.m)
else:
    # generate [0,L]^2 mesh via Firedrake
    mesh = RectangleMesh(args.m, args.m, L, L)
assert (not args.jaccard) or mesh.comm.size == 1, 'jaccard does not work in parallel'

# solver parameters
sp = {"snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-2,  # max u ~ 10^9, so roughly within 1 part in 10^-11 for u=H^{8/3}
    "snes_rtol": 1.0e-6,
    "snes_atol": 1.0e-10,
    "snes_stol": 1.0e-10, # FIXME??  why does it even matter?  in any case, keep it tight
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
    return - (1.0 / omega) * (u + 1.0)**phi * grad(b) # eps=1 regularization is small

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
    # mark and refine based on constraint u >= 0
    if i > 0:
        if args.jaccard:
            # generate active set indicator so we can evaluate Jaccard index
            eactive = amr.elemactive(u, lb)
        print('refining free boundary (VCD)', end='')
        # expand bracket vs default [0.2, 0.8], to provide high-res
        #   for ice near margin (0.8 -> 0.9) and to then accomodate
        #   advance into ice-free areas because the margin is resolved
        #   (0.2 -> 0.1)
        mark = amr.vcdmark(mesh, u, lb, bracket=[0.1, 0.9])
        if args.rmethod in ['always', 'alternate']:
            if args.rmethod == 'alternate' and i % 2 == 0:
                print(' and uniformly in inactive ...')
                imark = amr.eleminactive(u, lb)
            else:
                print(' and by grad recovery in inactive ...')
                imark, _ = amr.gradrecinactivemark(u, lb, theta=0.5)
            _, DG0 = amr.spaces(mesh)
            mark = Function(DG0).interpolate((mark + imark) - (mark * imark)) # union
        mesh = amr.refinemarkedelements(mesh, mark)

    # describe current mesh
    nv, ne, hmin, hmax = amr.meshsizes(mesh)
    print(f'solving problem {args.prob} on mesh level {i}:')
    amr.meshreport(mesh)

    # space for transformed thickness u
    V = FunctionSpace(mesh, "CG", 1)

    # bedrock and accumulation
    if args.data:
        b = Function(V).project(topg) # cross-mesh projection from data mesh
        # SMB from linear model based on lapse rate; from linearizing dome case
        c0 = -3.4e-8
        c1 = (6.3e-8 - c0) / 3.6e3
        a_lapse = c0 + c1 * topg
        a = Function(V).interpolate(conditional(nearb > 0.0, -1.0e-6, a_lapse)) # also cross-mesh re nearb
    else:
        x = SpatialCoordinate(mesh)
        if args.prob == 'dome':
            b = Function(V).interpolate(Constant(0.0))
            sexact = Function(V).interpolate(dome_exact(x))
            sexact.rename("s_exact")
        else:
            b = Function(V).interpolate(bumps(x, problem=args.prob))
        a = Function(V).interpolate(accumulation(x, problem=args.prob))
    b.rename("b = bedrock topography")
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
        #   note: u = H^(8/3) < 1 is *very* little ice in an initial iterate
        uold = Function(V).interpolate(conditional(uold < 1.0, 0.0, uold))

    # solve on current mesh
    u = Function(V, name="u = transformed thickness").interpolate(uold)
    lb = Function(V).interpolate(Constant(0.0))  # lower bound *in solver*
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    bcs = [DirichletBC(V, Constant(0.0), "on_boundary"),]
    if args.newton:
        F = weakform_primal(u, a, b)
        problem = NonlinearVariationalProblem(F, u, bcs=bcs)
        solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="s")
        solver.solve(bounds=(lb, ub))
    else:
        # outer loop for freeze iteration
        for k in range(args.freezecount):
            #print(f'  freeze tilt iteration {k+1} ...')
            Ztilt = Phi(uold, b)
            F = weakform_primal(u, a, b, Z=Ztilt)
            problem = NonlinearVariationalProblem(F, u, bcs=bcs)
            solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="s")
            solver.solve(bounds=(lb, ub))
            uold = Function(V).interpolate(u)

    # update true geometry variables
    H = Function(V, name="H = thickness").interpolate(u**omega)
    s = Function(V, name="s = surface elevation").interpolate(b + H)

    # report numerical errors if exact solution known
    if args.prob == 'dome':
        sdiff = Function(V).interpolate(s - dome_exact(x))
        sdiff.rename("sdiff = s - s_exact")
        err_l2 = norm(sdiff / L)
        err_av = norm(sdiff, 'l1') / L**2
        print('  |s-s_exact|_2 = %.3f m,  |s-s_exact|_av = %.3f m' \
                 % (err_l2, err_av))

    # report active set agreement between consecutive meshes using Jaccard index
    if i > 0 and args.jaccard:
        neweactive = amr.elemactive(u, lb)
        jac = amr.jaccard(eactive, neweactive)
        print(f'  glaciated areas Jaccard agreement {100*jac:.2f}% [levels {i-1}, {i}]' )
        eactive = neweactive

# save results from final mesh
if args.opvd:
    CU = ((n+2)/(n+1)) * Gamma
    U_ufl = CU * H**p * inner(grad(s), grad(s))**((p-2)/2) * grad(s)
    U = Function(VectorFunctionSpace(mesh, 'CG', degree=2))
    U.project(secpera * U_ufl)  # smoother than .interpolate()
    U.rename("U = surface velocity (m/a)")
    q = Function(FunctionSpace(mesh, 'BDM', 1))
    q.interpolate(U * H)
    q.rename("q = UH = *post-computed* ice flux")
    rank = Function(FunctionSpace(mesh, 'DG', 0))
    rank.dat.data[:] = mesh.comm.rank
    rank.rename('rank')
    print('writing to %s ...' % args.opvd)
    if args.prob == 'dome':
        VTKFile(args.opvd).write(a,u,s,H,U,q,sexact,sdiff,rank)
    else:
        Gb = Function(VectorFunctionSpace(mesh, 'DG', degree=0))
        Gb.interpolate(grad(b))
        Gb.rename('Gb = grad(b)')
        VTKFile(args.opvd).write(a,u,s,H,U,q,b,Gb,rank)
