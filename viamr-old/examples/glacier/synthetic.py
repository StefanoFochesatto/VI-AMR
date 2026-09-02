# Formulas for synthetic glacier examples.  The "dome" case with
# exact solution is here.  Also the bumpy bed formula used by
# steady.py with -prob cap|range

import numpy as np
from pyop2.mpi import MPI
import firedrake as fd
from firedrake.petsc import PETSc

# constants (same for all problems)
L = 1800.0e3  # domain is [0,L]^2, with fields centered at (xc,xc)
xc = L / 2
secpera = 31556926.0
n = 3.0
g = 9.81
rho = 910.0
A = 1.0e-16 / secpera
Gamma = 2 * A * (rho * g) ** n / (n + 2)

# dome parameters
domeL = 750.0e3
domeH0 = 3600.0


# exact solution to prob=='dome'
def dome_exact(x, n=3.0):
    # https://github.com/bueler/sia-fve/blob/master/petsc/base/exactsia.c#L83
    r = fd.sqrt(fd.dot(x - fd.as_vector([xc, xc]), x - fd.as_vector([xc, xc])))
    mm = 1 + 1 / n
    qq = n / (2 * n + 2)
    CC = domeH0 / (1 - 1 / n) ** qq
    z = r / domeL
    tmp = mm * z - 1 / n + (1 - z) ** mm - z**mm
    expr = CC * tmp**qq
    sexact = fd.conditional(fd.lt(r, domeL), expr, 0)
    return sexact


# accumulation; uses dome parameters
def accumulation(x, n=3.0, problem="cap"):
    # https://github.com/bueler/sia-fve/blob/master/petsc/base/exactsia.c#L51
    R = fd.sqrt(fd.dot(x - fd.as_vector([xc, xc]), x - fd.as_vector([xc, xc])))
    r = fd.conditional(fd.lt(R, 0.01), 0.01, R)
    r = fd.conditional(fd.gt(r, domeL - 0.01), domeL - 0.01, r)
    s = r / domeL
    C = domeH0 ** (2 * n + 2) * Gamma / (2 * domeL * (1 - 1 / n)) ** n
    pp = 1 / n
    tmp1 = s**pp + (1 - s) ** pp - 1
    tmp2 = 2 * s**pp + (1 - s) ** (pp - 1) * (1 - 2 * s) - 1
    a0 = (C / r) * tmp1 ** (n - 1) * tmp2
    if problem == "range":
        dxc = x[0] - xc
        dyc = x[1] - xc
        dd = L / 30
        aneg = -3.0e-8  # roughly the min of a0
        return fd.conditional(
            fd.gt(R, domeL),
            a0,
            fd.conditional(
                fd.lt(dxc**2, (1.9 * dd) ** 2),
                aneg,
                fd.conditional(fd.lt(dyc**2, (1.1 * dd) ** 2), aneg, a0),
            ),
        )
    else:
        return a0


def bumps(x, problem="cap"):
    if problem == "range":
        B0 = 400.0  # (m); amplitude of bumps
    else:
        B0 = 200.0  # (m); amplitude of bumps
    xx, yy = x[0] / L, x[1] / L
    b = (
        +5.0 * fd.sin(fd.pi * xx) * fd.sin(fd.pi * yy)
        + fd.sin(fd.pi * xx) * fd.sin(3 * fd.pi * yy)
        - fd.sin(2 * fd.pi * xx) * fd.sin(fd.pi * yy)
        + fd.sin(3 * fd.pi * xx) * fd.sin(3 * fd.pi * yy)
        + fd.sin(3 * fd.pi * xx) * fd.sin(5 * fd.pi * yy)
        + fd.sin(4 * fd.pi * xx) * fd.sin(4 * fd.pi * yy)
        - 0.5 * fd.sin(4 * fd.pi * xx) * fd.sin(5 * fd.pi * yy)
        - fd.sin(5 * fd.pi * xx) * fd.sin(2 * fd.pi * yy)
        - 0.5 * fd.sin(10 * fd.pi * xx) * fd.sin(10 * fd.pi * yy)
        + 0.5 * fd.sin(19 * fd.pi * xx) * fd.sin(11 * fd.pi * yy)
        + 0.5 * fd.sin(12 * fd.pi * xx) * fd.sin(17 * fd.pi * yy)
    )
    return B0 * b


def normerrorsdome(uh, Hh):
    """Return relative H^1 norm error in u and L^infty norm error in H.
    For the first, generate uexact in better space (and UFL from
    dome_exact()).  For L^infty error in H we merely want the max nodal
    error, so using V=CG1 is fine."""
    V = uh.function_space()
    x = fd.SpatialCoordinate(V.mesh())
    Hdiff = fd.Function(V).interpolate(Hh - dome_exact(x))
    Hdiff.rename("Hdiff = H - Hexact")
    with Hdiff.dat.vec_ro as v:
        Herr = abs(v).max()[1]
    CG2 = fd.FunctionSpace(V.mesh(), "CG", 2)
    p = n + 1  # typical:  p = 4
    omega = (p - 1) / (2 * p)  #  omega = 3/8
    uexact = fd.Function(CG2).interpolate(dome_exact(x) ** (1.0 / omega))
    uexact.rename("uexact")
    uerr = fd.errornorm(uexact, uh, norm_type="H1") / fd.norm(uexact, norm_type="H1")
    return uerr, Herr


def radiuserrordome(mesh, vfb):
    """For -prob "dome", compute the maximum of free-boundary radius errors
    from the output of VIAMR.freeboundarygraph().  The exact free boundary
    is a circle of radius domeL with center (L/2,L/2).  Returns the maximum
    radius error."""
    vfb = np.array(vfb)
    mymax = PETSc.NINFINITY
    if len(vfb) > 0:
        x, y = vfb[:, 0], vfb[:, 1]
        drfb = np.abs(np.sqrt((x - L / 2) ** 2 + (y - L / 2) ** 2) - domeL)
        mymax = np.max(drfb)
    drmax = float(mesh.comm.allreduce(mymax, op=MPI.MAX))
    return drmax
