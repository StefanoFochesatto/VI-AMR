from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(description="""
Solves 2D steady shallow ice approximation obstacle problem on a square [0,L]^2
where L = 1800.0 km.  Default behavior is to solve for the "dome" exact
solution, on a flat bed.  Optionally (-bumps) generates a random, but smooth,
bed topography; there is no exact solution in this case.  Note there is a double
regularization in the isothermal, Glen-law diffusivity; see options -epsH and -epsplap.
Default solver is vinewtonrsls + mumps, iterated to convergence.
Example runs with achievable resolution:
  python3 glacier.py -m 80 -opvd dome.pvd
  python3 glacier.py -m 80 -bumps -opvd bumps.pvd
""", formatter_class=RawTextHelpFormatter)
parser.add_argument('-bumps', action='store_true', default=False,
                    help='generate bumpy bed topography')
parser.add_argument('-epsH', type=float, default=20.0, metavar='X',
                    help='diffusivity regularization for thickness [default 20.0 m]')
parser.add_argument('-epsplap', type=float, default=1.0e-4, metavar='X',
                    help='diffusivity regularization for p-Laplacian [default 1.0e-4]')
parser.add_argument('-m', type=int, default=20, metavar='MC',
                    help='number of cells in each direction [default=20]')
parser.add_argument('-opvd', metavar='FILE', type=str, default='',
                    help='output file name for Paraview format (.pvd)')
parser.add_argument('-qdegree', type=int, default=4, metavar='DEG',
                    help='quadrature degree used in SIA nonlinear weak form [default 4]')
args, passthroughoptions = parser.parse_known_args()

assert args.m >= 1, 'at least one cell in mesh'

import petsc4py
petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.output import VTKFile

L = 1800.0e3        # domain is [0,L]^2, with fields centered at (xc,xc)
xc = L/2
secpera = 31556926.0
n = Constant(3.0)
p = n + 1
A = Constant(1.0e-16) / secpera
g = Constant(9.81)
rho = Constant(910.0)
Gamma = 2*A*(rho * g)**n / (n+2)

domeL = 750.0e3
domeH0 = 3600.0

def accumulation(x):
    # https://github.com/bueler/sia-fve/blob/master/petsc/base/exactsia.c#L51
    R = sqrt(dot(x - as_vector([xc, xc]), x - as_vector([xc, xc])))
    r = conditional(lt(R, 0.01), 0.01, R)
    r = conditional(gt(r, domeL - 0.01), domeL - 0.01, r)
    s = r / domeL
    C = domeH0**(2*n + 2) * Gamma / (2 * domeL * (1 - 1/n) )**n
    pp = 1/n
    tmp1 = s**pp + (1-s)**pp - 1
    tmp2 = 2*s**pp + (1-s)**(pp-1) * (1-2*s) - 1
    a = (C / r) * tmp1**(n-1) * tmp2
    return a

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

def bumps(x):
    B0 = 200.0  # (m); amplitude of bumps
    xx, yy = x[0] / L, x[1] / L
    b = + 5.0 * sin(pi*xx) * sin(pi*yy) \
        + sin(pi*xx) * sin(3*pi*yy) - sin(2*pi*xx) * sin(pi*yy) \
        + sin(3*pi*xx) * sin(3*pi*yy) + sin(3*pi*xx) * sin(5*pi*yy) \
        + sin(4*pi*xx) * sin(4*pi*yy) - 0.5 * sin(4*pi*xx) * sin(5*pi*yy) \
        - sin(5*pi*xx) * sin(2*pi*yy) - 0.5 * sin(10*pi*xx) * sin(10*pi*yy) \
        + 0.5 * sin(19*pi*xx) * sin(11*pi*yy) + 0.5 * sin(12*pi*xx) * sin(17*pi*yy)
    return B0 * b

mesh = RectangleMesh(args.m, args.m, L, L)
x = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)
s = Function(V, name="s = surface elevation")  # initialize to zero
a = Function(V).interpolate(accumulation(x))
a.rename("a = accumulation")

if args.bumps:
    lb = Function(V).interpolate(bumps(x))
    lb.rename("lb = bedrock topography")
else:
    lb = Function(V).interpolate(Constant(0))
    sexact = Function(V).interpolate(dome_exact(x))
    sexact.rename("s_exact")

v = TestFunction(V)
Dplap = (args.epsplap**2 + dot(grad(s), grad(s)))**((p-2)/2)
F = Gamma * (s - lb + args.epsH)**(p+1) * Dplap * inner(grad(s), grad(v)) * dx(degree=args.qdegree) \
    - inner(a, v) * dx
bcs = DirichletBC(V, 0, "on_boundary")
problem = NonlinearVariationalProblem(F, s, bcs)

# initialize to pile of ice, from accumulation
pilefactor = 400.0
s.interpolate(lb + pilefactor * secpera * conditional(a > 0.0, a, 0.0))

sp = {"snes_type": "vinewtonrsls",
    "snes_rtol": 1.0e-4,  # low regularity implies lowered expectations
    "snes_atol": 1.0e-12,
    "snes_stol": 1.0e-5,
    "snes_monitor": None,
    "snes_linesearch_type": "bt",
    "snes_linesearch_order": "1",
    "snes_max_it": 1000,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"}
solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="")
solver.solve(bounds=(lb, Function(V).interpolate(Constant(PETSc.INFINITY))))

from firedrake.petsc import PETSc
PETSc.Sys.Print('done (%s): %d x %d mesh, h=%.3f km' % \
                ('bumps' if args.bumps else 'dome', args.m, args.m, L / (1000 * args.m)))

if not args.bumps:
    sdiff = Function(V).interpolate(s - sexact)
    sdiff.rename("sdiff = s - s_exact")
    with sdiff.dat.vec_ro as v:
        err_max = abs(v).max()[1]
    err_av = assemble(abs(sdiff) * dx) / L**2
    PETSc.Sys.Print('             |s-s_exact|_inf = %.3f m,  |s-s_exact|_av = %.3f m' \
                    % (err_max, err_av))

if args.opvd:
    CU = ((n+2)/(n+1)) * Gamma
    U_ufl = CU * (s - lb)**p * inner(grad(s), grad(s))**((p-2)/2) * grad(s)
    U = Function(VectorFunctionSpace(mesh, 'CG', degree=2))
    U.project(secpera * U_ufl)
    U.rename("U = surface velocity (m/a)")
    Q = Function(VectorFunctionSpace(mesh, 'CG', degree=1))
    Q.interpolate(U * (s - lb))
    Q.rename("Q = UH = volume flux (m^2/a)")
    PETSc.Sys.Print('writing to %s ...' % args.opvd)
    if args.bumps:
        VTKFile(args.opvd).write(a,s,U,Q,lb)
    else:
        VTKFile(args.opvd).write(a,s,U,Q,sexact,sdiff)
