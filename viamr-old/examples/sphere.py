# This example attempts to do apples-to-apples comparisons of all three
# algorithms on the "ball" problem, where the exact solution is known
# and we can compute norm convergence rates.  Uniform refinement is also done.
# We generate four .pvd files, result_sphere_{udo,vcd,avm,uniform}.pvd, suitable
# for a figure in the paper.  Optionally we generate .csv files for norm and
# Jaccard convergence rates.  Note we are comparing n=1 UDO to default [0.2,0.8]
# VCD.

import time
import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
from viamr import VIAMR

try:
    import netgen
except ImportError:
    raise ImportError("Unable to import NetGen.  Exiting.")
from netgen.geom2d import SplineGeometry

print = PETSc.Sys.Print  # enables correct printing in parallel

# number of AMR refinements; use e.g. levels = 11, and parallel, for serious convergence
# generally uniform can't reach high levels; suggest  uniformlevels = 0.6 levels,
# e.g. levels=11 --> uniformlevels=7
m0 = 12           # for UDO,VCD,UNI initial mesh is m0 x m0; see below for AVM
levels = 4
uniformlevels = 4
writecsvs = False

# method parameters
thetaBR = 0.4  # controls BR resolution in inactive set, and convergence rate

# AVM parameters; attempts to do apples-to-apples vs UDO|VCD+BR
initialhAVM = 4.0 / m0
targetsAVM = [100, 300, 900, 3000, 7000, 18000, 50000, 100000, 250000, 600000, 1400000, 3000000]


def psiUFL(r):
    """obstacle as UFL, from UFL expression for r"""
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def uexactUFL(r):
    """exact solution (and boundary conditions) as UFL, from UFL expression for r"""
    afree = 0.697965148223374
    A, B = 0.680259411891719, 0.471519893402112
    return conditional(le(r, afree), psiUFL(r), -A * ln(r) + B)


def activeexactUFL(r):
    """exact active set as UFL, from UFL expression for r"""
    afree = 0.697965148223374
    return conditional(le(r, afree), 1.0, 0.0)


def errornormpreferred(r, uh, activeh):
    """error norm against "preferred" form of numerical solution"""
    tildeuh = conditional(eq(activeh, 1.0), psiUFL(r), uh)  # preferred
    # high degree quadrature important in next line
    normsq = assemble((uexactUFL(r) - tildeuh) ** 2 * dx)
    return np.sqrt(normsq)


# solver parameters for VI
sp = {
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-12,
    "snes_linesearch_type": "basic",
    "snes_max_it": 200,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-12,
    "snes_stol": 0.0,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    # "snes_vi_monitor": None,
    "snes_converged_reason": None,
}

for amrtype in ["udo", "vcd", "uni", "avm"]:
    methodname = amrtype.upper()
    if methodname != "AVM" and methodname != "UNI":
        methodname += "+BR"
    print(f"solving by VIAMR using {methodname} method ...")

    if writecsvs:
        csvfile = open(f"sphere_{methodname}.csv", "w")
        csvfile.write("I,NV,NE,HMIN,HMAX,ENORM,ENORMPREF,JACCARD,REFINETIME\n")

    amr = VIAMR()

    # setting distribution parameters should not be necessary ... but bug in netgen
    dp = {
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
    }
    if amrtype == "avm":
        geo = SplineGeometry()
        geo.AddRectangle(p1=(-2, -2), p2=(2, 2), bc="rectangle")
        ngmsh = geo.GenerateMesh(maxh=initialhAVM)
        mesh0 = Mesh(ngmsh, distribution_parameters=dp)
    else:
        mesh0 = RectangleMesh(
            m0,
            m0,
            Lx=2.0,
            Ly=2.0,
            originX=-2.0,
            originY=-2.0,
            distribution_parameters=dp,
        )
    meshHist = [mesh0]

    if amrtype == "uni":
        unimh = MeshHierarchy(mesh0, uniformlevels)

    for i in range(levels + 1):
        mesh = meshHist[i]
        x, y = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        print(f"solving on mesh {i} ...")
        amr.meshreport(mesh)

        V = FunctionSpace(mesh, "CG", 1)
        if i == 0:
            uh = Function(V, name="u_h")
            refinetime = 0.0
        else:
            # initialize by cross-mesh interpolation to fine mesh
            uUFL = conditional(uh < lb, lb, uh)  # use old data
            uh = Function(V, name="u_h").interpolate(uUFL)

        v = TestFunction(V)
        F = inner(grad(uh), grad(v)) * dx
        bcs = DirichletBC(V, uexactUFL(r), "on_boundary")
        problem = NonlinearVariationalProblem(F, uh, bcs)

        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="s"
        )
        lb = Function(V, name="psi").interpolate(psiUFL(r))
        ub = Function(V).interpolate(Constant(PETSc.INFINITY))
        solver.solve(bounds=(lb, ub))

        en_no = errornorm(uexactUFL(r), uh)
        activeh = amr.elemactive(uh, lb)
        en_pre = errornormpreferred(r, uh, activeh)
        print(f"  ||u_exact - u_h||_2 = {en_no:.3e}")
        print(f"  ||u_exact - tilde u_h||_2 = {en_pre:.3e}")

        jaccard = amr.jaccardUFL(activeexactUFL(r), activeh)
        print(f"  jaccard(A_u, A_uh) = {jaccard:.5f}")

        if writecsvs:
            Nv, Ne, hmin, hmax = amr.meshsizes(mesh)
            csvfile.write(
                f"{i},{Nv},{Ne},{hmin:.5f},{hmax:.5f},{en_no:.3e},{en_pre:.3e},{jaccard:.5f},{refinetime:.3e}\n"
            )

        if amrtype == "uni" and i >= uniformlevels:
            break

        if i >= levels:
            break

        start_time = time.time()
        if amrtype == "uni":
            mesh = unimh[i+1]
        elif amrtype == "avm":
            amr.setmetricparameters(target_complexity=targetsAVM[i+1], h_min=1.0e-4, h_max=1.0)
            mesh = amr.adaptaveragedmetric(mesh, uh, lb)
        else:
            if amrtype == "udo":
                mark = amr.udomark(uh, lb, n=1)
            elif amrtype == "vcd":
                mark = amr.vcdmark(uh, lb)
            residual = -div(grad(uh))
            (imark, _, _) = amr.brinactivemark(uh, lb, residual, theta=thetaBR)
            mark = amr.unionmarks(mark, imark)
            mesh = amr.refinemarkedelements(mesh, mark)
        if amrtype != "uni":
            refinetime = time.time() - start_time

        meshHist.append(mesh)

    if writecsvs:
        csvfile.close()

    outfile = "result_sphere_" + amrtype + ".pvd"
    print(f"done ... writing to {outfile} ...")
    V = uh.function_space()
    gap = Function(V, name="gap = u_h - lb").interpolate(uh - lb)
    uexactint = Function(V, name="u_exact").interpolate(uexactUFL(r))
    error = Function(V, name="error = |pi_h(u_exact) - u_h|").interpolate(
        abs(uexactint - uh)
    )
    VTKFile(outfile).write(uh, lb, gap, uexactint, error)
    print("")
