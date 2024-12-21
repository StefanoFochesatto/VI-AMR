# TODO * allow other strategies than alternating AMR & uniform?
#      * fix -data so it works

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
Solver is vinewtonrsls + mumps.
Applies VIAMR VCES method because UD0 is not currently parallel.
Refinement schedule is alternation with uniform refinement.
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
parser.add_argument('-refine', type=int, default=2, metavar='R',
                    help='number of AMR refinements [default 2]')
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
    if args.data:
        raise NotImplementedError('incompatible arguments -onelevel -data')
    args.refine = 0
    printpar('not using refinement; uniform mesh generated with Firedrake')

# read data for bed topography into numpy array
#     (assumes fixed variable names 'x1', 'y1', 'topg'
if args.data:
    import netCDF4
    data = netCDF4.Dataset(args.data)
    data.set_auto_mask(False)  # otherwise irritating masked arrays
    topg_np = data.variables['topg'][0,:,:].T  # transpose immediately
    x_np = data.variables['x1']
    y_np = data.variables['y1']
    llxy = (min(x_np), min(y_np))  # lower left
    urxy = (max(x_np), max(y_np))  # upper right
    llstr = f'({llxy[0]/1000.0:.3f},{llxy[1]/1000.0:.3f})'
    urstr = f'({urxy[0]/1000.0:.3f},{urxy[1]/1000.0:.3f})'
    mx_np, my_np = np.shape(topg_np)
    hx_np, hy_np = x_np[1] - x_np[0], y_np[1] - y_np[0]
    printpar(f'reading from NetCDF file {args.data} ...')
    printpar(f'    rectangle {llstr}-->{urstr} km')
    printpar(f'    {mx_np} x {my_np} grid with {hx_np/1000.0:.3f} x {hx_np/1000.0:.3f} km spacing')
    if False:  # debug view of data if True
        import matplotlib.pyplot as plt
        plt.pcolormesh(x_np, y_np, topg_np)
        plt.axis('equal')
        plt.show()
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
    if args.data:
        geo.AddRectangle(p1=llxy, p2=urxy, bc="rectangle")
        trih = max(urxy[0] - llxy[0], urxy[1] - llxy[1]) / args.m
    else:
        geo.AddRectangle(p1=(0.0, 0.0), p2=(L, L), bc="rectangle")
        trih = L / args.m
    ngmsh = None
    ngmsh = geo.GenerateMesh(maxh=trih)
    mesh = Mesh(ngmsh)

if args.data:
    # read structured-mesh data into Firedrake data mesh
    printpar(f'    reading "topg" into Q1 data mesh ...')
    dmesh = RectangleMesh(mx_np - 1,
                          my_np - 1,
                          urxy[0] - llxy[0],
                          urxy[1] - llxy[1])
    dmesh.coordinates.dat.data[:,0] += llxy[0]
    dmesh.coordinates.dat.data[:,1] += llxy[1]
    dCG1 = FunctionSpace(dmesh, "CG", 1)
    topg = Function(dCG1)
    nearb = Function(dCG1)  # set to zero here
    delnb = 100.0e3  # within this far of boundary, will apply negative SMB
    for k in range(len(topg.dat.data)):
        xk, yk = dmesh.coordinates.dat.data[k]
        i = int((xk - llxy[0]) / hx_np)
        j = int((yk - llxy[1]) / hy_np)
        topg.dat.data[k] = topg_np[i][j]
        db = min([abs(xk - llxy[0]), abs(xk - urxy[0]), abs(yk - llxy[1]), abs(yk - urxy[1])])
        if db < delnb:
            nearb.dat.data[k] = 1.0
    #VTKFile('topg.pvd').write(topg, nearb)

# solver parameters
sp = {"snes_type": "vinewtonrsls",
    "snes_rtol": 1.0e-4,  # low regularity implies lowered expectations
    "snes_atol": 1.0e-8,
    "snes_stol": 1.0e-5,
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_linesearch_type": "bt",
    "snes_linesearch_order": "1",
    "snes_max_it": 1000,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"}

# transformed SIA
p = n + 1
omega = (p - 1) / (2*p)
phi = (p + 1) / (2*p)

def Phi(u, b):
    return - (1.0 / omega) * u**phi * grad(b)  # FIXME consider further softening grad(b) if real beds are a problem

def tranformedweakform(u, v, a, Z):
    dstilt = grad(u) - Z
    Dp = inner(dstilt, dstilt)**((p-2)/2)
    return Gamma * Dp * inner(dstilt, grad(v)) * dx - a * v * dx

# main loop
for i in range(args.refine + 1):
    if i > 0 and args.jaccard:
        # generate active set indicator so we can evaluate Jaccard index
        eactive = VIAMR(debug=True).elemactive(s, lb)

    if i > 0:
        # refinement schedule is to alternate: uniform,AMR,uniform,AMR,uniform,...
        if np.mod(i, 2) == 0:
            mark = VIAMR().vcesmark(mesh, s, lb)
            #mark = VIAMR().udomark(mesh, s, lb, n=1)  # not in parallel, but otherwise great
        else:
            W = FunctionSpace(mesh, "DG", 0)
            mark = Function(W).interpolate(Constant(1.0))  # mark everybody
        mesh = mesh.refine_marked_elements(mark)  # NetGen gives next mesh

    V = FunctionSpace(mesh, "CG", 1)

    # obstacle and source term
    if args.data:
        lb = Function(V).project(topg) # cross-mesh projection from data mesh
        # SMB from linear model based on lapse rate; from linearizing dome case
        c0 = -3.4e-8
        #c1 = (6.3e-8 - c0) / 3.6e3
        c1 = (4.5e-8 - c0) / 3.6e3
        a_lapse = c0 + c1 * topg
        a = Function(V).interpolate(conditional(nearb > 0.0, -1.0e-6, a_lapse))
        #VTKFile('data.pvd').write(lb, a)
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
    u = Function(V, name="u = transformed thickness").interpolate(uold)

    # Picard iterate the tilted p-Laplacian problem
    nv, ne, hmin, hmax = VIAMR().meshsizes(mesh)
    printpar(f'solving level {i}: {nv} vertices, {ne} elements, h in [{hmin/1e3:.3f},{hmax/1e3:.3f}] km ...')
    v = TestFunction(V)
    bcs = DirichletBC(V, Constant(0.0), "on_boundary")
    for k in range(4):
        printpar(f'    Picard iteration {k} ...')
        Z = Phi(uold, lb)
        F = tranformedweakform(u, v, a, Z)
        problem = NonlinearVariationalProblem(F, u, bcs)
        solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="")
        lower = Function(V).interpolate(Constant(0.0))
        upper = Function(V).interpolate(Constant(PETSc.INFINITY))
        solver.solve(bounds=(lower, upper))
        uold = Function(V).interpolate(u)
    H = Function(V, name="H = thickness").interpolate(u**omega)
    s = Function(V, name="s = surface elevation").interpolate(lb + H)

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
        VTKFile(args.opvd).write(a,u,s,H,U,Q,sexact,sdiff)
    else:
        VTKFile(args.opvd).write(a,u,s,H,U,Q,lb)
