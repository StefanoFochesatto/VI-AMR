# TODO * allow other strategies than alternating AMR & uniform?
#      * read Greenland data from a .nc file

from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(description="""
Solves 2D steady shallow ice approximation glacier obstacle problem.
Synthetic problem (default):
    Domain is a square [0,L]^2 where L = 1800.0 km. By default generates
    a random, but smooth, bed topography.  Option -prob dome solves in
    a flat bed case where the exact solution is known, and -prob range
    generates a disconnected glacier.
Data-based problem (-data DATA.nc):
    FIXME WIP
Note there is a double regularization in the isothermal, Glen-law diffusivity;
see options -epsH and -epsplap.  Solver is vinewtonrsls + mumps.  Applies VIAMR
VCES method (because UD0 is not currently parallel) alternately with uniform
refinement.
Examples:
  python3 steady.py -opvd cap.pvd
  python3 steady.py -prob dome -opvd dome.pvd
Works in serial only:
  python3 steady.py -jaccard
Also runs in parallel:
  mpiexec -n 4 steady.py -opvd cap4.pvd
""", formatter_class=RawTextHelpFormatter)
parser.add_argument('-data', metavar='FILE', type=str, default='',
                    help='read "topg" variable from NetCDF file (.nc)')
parser.add_argument('-epsH', type=float, default=20.0, metavar='X',
                    help='diffusivity regularization for thickness [default 20.0 m]')
parser.add_argument('-epsplap', type=float, default=1.0e-4, metavar='X',
                    help='diffusivity regularization for p-Laplacian [default 1.0e-4]')
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
parser.add_argument('-qdegree', type=int, default=4, metavar='DEG',
                    help='quadrature degree used in SIA nonlinear weak form [default 4]')
parser.add_argument('-refine', type=int, default=3, metavar='R',
                    help='number of AMR refinements [default 3]')
args, passthroughoptions = parser.parse_known_args()

import numpy as np
import petsc4py
petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
printpar = PETSc.Sys.Print
from viamr import VIAMR

assert args.m >= 1, 'at least one cell in mesh'
if args.onelevel:
    args.refine = 0
    printpar('not using refinement; uniform mesh generated with Firedrake')

# read data for bed topography into numpy array, from 'topg' variable
if args.data:
    import netCDF4
    data = netCDF4.Dataset(args.data)
    data.set_auto_mask(False)  # otherwise irritating masked arrays
    lb_np = data.variables['topg'][0,:,:]
    if False:  # debug view of data if True
        import matplotlib.pyplot as plt
        x_d = data.variables['x1']
        y_d = data.variables['y1']
        plt.pcolormesh(x_d, y_d, lb_np)
        plt.axis('equal')
        plt.show()

# constants (same for all problems)
L = 1800.0e3        # domain is [0,L]^2, with fields centered at (xc,xc)
xc = L/2
secpera = 31556926.0
n = Constant(3.0)
p = n + 1
g = Constant(9.81)
rho = Constant(910.0)
A = Constant(1.0e-16) / secpera
Gamma = 2*A*(rho * g)**n / (n+2)
aGamma = Gamma   # used in accumulation
if args.prob == 'range':
    Gamma *= 10.0

# exact solution to prob=='dome'
domeL = 750.0e3
domeH0 = 3600.0
def dome_exact(x):
    # https://github.com/bueler/sia-fve/blob/master/petsc/base/exactsia.c#L83
    r = sqrt(dot(x - as_vector([xc, xc]), x - as_vector([xc, xc])))
    mm = 1 + 1/n
    qq = n / (2*n + 2)
    CC = domeH0 / (1-1/n)**qq
    z = r / domeL
    tmp = mm * z - 1/n + (1-z)**mm - z**mm
    expr = CC * tmp**qq
    sexact = conditional(lt(r, domeL), expr, 0)
    return sexact

# accumulation; uses dome parameters
def accumulation(x, problem='cap'):
    # https://github.com/bueler/sia-fve/blob/master/petsc/base/exactsia.c#L51
    R = sqrt(dot(x - as_vector([xc, xc]), x - as_vector([xc, xc])))
    r = conditional(lt(R, 0.01), 0.01, R)
    r = conditional(gt(r, domeL - 0.01), domeL - 0.01, r)
    s = r / domeL
    C = domeH0**(2*n + 2) * aGamma / (2 * domeL * (1 - 1/n) )**n
    pp = 1/n
    tmp1 = s**pp + (1-s)**pp - 1
    tmp2 = 2*s**pp + (1-s)**(pp-1) * (1-2*s) - 1
    a0 = (C / r) * tmp1**(n-1) * tmp2
    if problem == 'range':
        dxc = x[0] - xc
        dyc = x[1] - xc
        dd = L / 30
        aneg = -3.0e-8  # roughly the min of a0
        return conditional(gt(R, domeL), a0,
                           conditional(lt(dxc**2, (1.9 * dd)**2), aneg,
                                       conditional(lt(dyc**2, (1.1 * dd)**2), aneg, a0)))
    else:
        return a0

def bumps(x, problem='cap'):
    if problem == 'range':
        B0 = 400.0  # (m); amplitude of bumps
    else:
        B0 = 200.0  # (m); amplitude of bumps
    xx, yy = x[0] / L, x[1] / L
    b = + 5.0 * sin(pi*xx) * sin(pi*yy) \
        + sin(pi*xx) * sin(3*pi*yy) - sin(2*pi*xx) * sin(pi*yy) \
        + sin(3*pi*xx) * sin(3*pi*yy) + sin(3*pi*xx) * sin(5*pi*yy) \
        + sin(4*pi*xx) * sin(4*pi*yy) - 0.5 * sin(4*pi*xx) * sin(5*pi*yy) \
        - sin(5*pi*xx) * sin(2*pi*yy) - 0.5 * sin(10*pi*xx) * sin(10*pi*yy) \
        + 0.5 * sin(19*pi*xx) * sin(11*pi*yy) + 0.5 * sin(12*pi*xx) * sin(17*pi*yy)
    return B0 * b

# generate first mesh
if args.data:
    # FIXME apparently I have to use the technique at
    #    https://docu.ngsolve.org/latest/netgen_tutorials/manual_mesh_generation.html
    import sys
    sys.exit(0)
else:
    printpar(f'generating synthetic {args.m} x {args.m} initial mesh for problem {args.prob} ...')
    if args.onelevel:
        # generate via Firedrake
        mesh = RectangleMesh(args.m, args.m, L, L)
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
        ngmsh = None
        trih = L / args.m
        ngmsh = geo.GenerateMesh(maxh=trih)
        mesh = Mesh(ngmsh)

# solver parameters
sp = {"snes_type": "vinewtonrsls",
    "snes_rtol": 1.0e-4,  # low regularity implies lowered expectations
    "snes_atol": 1.0e-12,
    "snes_stol": 1.0e-5,
    #"snes_monitor": None,
    "snes_converged_reason": None,
    "snes_linesearch_type": "bt",
    "snes_linesearch_order": "1",
    "snes_max_it": 1000,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"}

# main loop
for i in range(args.refine + 1):
    if i > 0 and args.jaccard:
        # generate active set indicator so we can evaluate Jaccard index
        eactive = VIAMR(debug=True).elemactive(s, lb)
    if i > 0:
        # refinement schedule is to alternate: AMR,uniform,AMR,uniform,...
        if np.mod(i, 2) == 1:
            mark = VIAMR().vcesmark(mesh, s, lb)
            #mark = VIAMR().udomark(mesh, s, lb, n=1)  # not in parallel, but otherwise great
        else:
            # mark everybody
            W = FunctionSpace(mesh, "DG", 0)
            mark = Function(W).interpolate(Constant(1.0))
        # use NetGen to get next mesh
        mesh = mesh.refine_marked_elements(mark)

    # obstacle and source term
    V = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)
    if args.data:
        # FIXME cross-mesh interpolate lb_d to lb on current mesh
        # FIXME plan is to compute a(x,y) from lb(x,y) using lapse rates
        a = Function(V).interpolate(Constant(0))
    else:
        if args.prob == 'dome':
            lb = Function(V).interpolate(Constant(0))
            sexact = Function(V).interpolate(dome_exact(x))
            sexact.rename("s_exact")
        else:
            lb = Function(V).interpolate(bumps(x, problem=args.prob))
        a = Function(V).interpolate(accumulation(x, problem=args.prob))
    lb.rename("lb = bedrock topography")
    a.rename("a = accumulation")

    if i == 0:
        # initialize as pile of ice, from accumulation
        pileage = 400.0  # years
        sinit = lb + pileage * secpera * conditional(a > 0.0, a, 0.0)
        s = Function(V).interpolate(sinit)
    else:
        # cross-mesh interpolation of previous solution
        s = Function(V).interpolate(s)
    s.rename("s = surface elevation")

    # weak form
    v = TestFunction(V)
    Hreg = s - lb + args.epsH
    Dplap = (args.epsplap**2 + dot(grad(s), grad(s)))**((p-2)/2)
    F = Gamma * Hreg**(p+1) * Dplap * inner(grad(s), grad(v)) * dx(degree=args.qdegree) \
        - inner(a, v) * dx
    bcs = DirichletBC(V, 0, "on_boundary")
    problem = NonlinearVariationalProblem(F, s, bcs)

    # solve
    nv, ne, hmin, hmax = VIAMR().meshsizes(mesh)
    printpar(f'solving level {i}, {nv} vertices, {ne} elements, h in [{hmin/1e3:.3f},{hmax/1e3:.3f}] km ...')
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="")
    solver.solve(bounds=(lb, Function(V).interpolate(Constant(PETSc.INFINITY))))

    if i > 0 and args.jaccard:
        z = VIAMR(debug=True)
        neweactive = z.elemactive(s, lb)
        jac = z.jaccard(eactive, neweactive)
        printpar(f'  glaciated areas Jaccard agreement {100*jac:.2f}% [levels {i-1}, {i}]' )
        eactive = neweactive

    if args.prob == 'dome':
        sdiff = Function(V).interpolate(s - dome_exact(x))
        sdiff.rename("sdiff = s - s_exact")
        err_l2 = norm(sdiff / L)
        err_av = norm(sdiff, 'l1') / L**2
        printpar('    |s-s_exact|_2 = %.3f m,  |s-s_exact|_av = %.3f m' \
                 % (err_l2, err_av))

if args.opvd:
    H = Function(V, name="H = thickness (m)").interpolate(s - lb)
    CU = ((n+2)/(n+1)) * Gamma
    U_ufl = CU * (s - lb)**p * inner(grad(s), grad(s))**((p-2)/2) * grad(s)
    U = Function(VectorFunctionSpace(mesh, 'CG', degree=2))
    U.project(secpera * U_ufl)  # smoother than .interpolate()
    U.rename("U = surface velocity (m/a)")
    Q = Function(VectorFunctionSpace(mesh, 'DG', degree=1))
    Q.interpolate(U * (s - lb))
    Q.rename("Q = UH = volume flux (m^2/a)")
    printpar('writing to %s ...' % args.opvd)
    if args.prob == 'dome':
        VTKFile(args.opvd).write(a,s,H,U,Q,sexact,sdiff)
    else:
        VTKFile(args.opvd).write(a,s,H,U,Q,lb)
