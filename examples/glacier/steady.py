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

from synthetic import secpera, n, Gamma, L, dome_exact, accumulation, bumps

assert args.m >= 1, 'at least one cell in mesh'
if args.onelevel:
    args.refine = 0
    printpar('not using refinement; uniform mesh generated with Firedrake')

# read data for bed topography into numpy array
#     (assumes fixed variable names 'x1', 'y1', 'topg'
if args.data:
    import netCDF4
    data = netCDF4.Dataset(args.data)
    data.set_auto_mask(False)  # otherwise irritating masked arrays
    lb_np = data.variables['topg'][0,:,:]
    x_np = data.variables['x1']
    y_np = data.variables['y1']
    if False:  # debug view of data if True
        import matplotlib.pyplot as plt
        plt.pcolormesh(x_np, y_np, lb_np)
        plt.axis('equal')
        plt.show()

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
    p = n + 1
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
