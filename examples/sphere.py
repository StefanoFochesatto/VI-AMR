# This example attempts to do apples-to-apples comparisons of all three
# algorithms on the "ball" problem, where the exact solution is known
# and we can compute norm convergence rates.  We generate three .pvd
# files, result_sphere_{udo,vcd,avm}.pvd, suitable for a figure in the
# paper.  Note we are comparing n=1 UDO to default [0.2,0.8] VCD on the
# sphere problem, for which an exact solution is known.

import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
from viamr import VIAMR

try:
    import netgen
except ImportError:
    raise ImportError("Unable to import NetGen.  Exiting.")
from netgen.geom2d import SplineGeometry

levels = 3  # number of AMR refinements; use e.g. levels = 7 for more serious convergence
m0 = 20  # for UDO and VCD, initial mesh is m0 x m0
initialhAVM = 4.0 / m0  # for apples-to-apples
targetAVM = 2000  # adjust to make apples-to-apples ish

# set-up for boundary conditions and exact solution
afree = 0.697965148223374
A, B = 0.680259411891719, 0.471519893402112
r0 = 0.9
psi0 = np.sqrt(1.0 - r0 * r0)
dpsi0 = -r0 / psi0

print = PETSc.Sys.Print  # enables correct printing in parallel

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

for amrtype in ["udo", "vcd", "avm"]:
    methodname = amrtype.upper()
    if methodname != "AVM":
        methodname += "+BR"
    print(f"solving by VIAMR using {methodname} method ...")
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
        amr.setmetricparameters(target_complexity=targetAVM, h_min=1.0e-4, h_max=1.0)
    else:
        mesh0 = RectangleMesh(m0, m0, Lx=2.0, Ly=2.0, originX=-2.0, originY=-2.0, distribution_parameters=dp)
    meshHist = [mesh0]

    for i in range(levels + 1):
        mesh = meshHist[i]
        print(f"solving on mesh {i} ...")
        amr.meshreport(mesh)

        x, y = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        obsUFL = conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))
        bdryUFL = conditional(le(r, afree), obsUFL, -A * ln(r) + B)

        V = FunctionSpace(mesh, "CG", 1)
        if i == 0:
            uh = Function(V, name="u_h")
        else:
            # initialize by cross-mesh interpolation to fine mesh
            uUFL = conditional(uh < lb, lb, uh)  # use old data
            uh = Function(V, name="u_h").interpolate(uUFL)

        v = TestFunction(V)
        F = inner(grad(uh), grad(v)) * dx
        bcs = DirichletBC(V, bdryUFL, "on_boundary")
        problem = NonlinearVariationalProblem(F, uh, bcs)

        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="s"
        )
        lb = Function(V, name="psi").interpolate(obsUFL)
        ub = Function(V).interpolate(Constant(PETSc.INFINITY))
        solver.solve(bounds=(lb, ub))

        uexact = Function(V, name="u_exact").interpolate(bdryUFL)
        print(f"  ||u_exact - u_h||_2 = {errornorm(bdryUFL, uh):.3e},  ||pi_h(u_exact) - u_h||_2 = {errornorm(uexact, uh):.3e}")

        if i == levels:
            break

        if amrtype == "avm":
            mesh = amr.adaptaveragedmetric(mesh, uh, lb)
        else:
            if amrtype == "udo":
                mark = amr.udomark(uh, lb, n=1)
            elif amrtype == "vcd":
                mark = amr.vcdmark(uh, lb)
            residual = -div(grad(uh))
            (imark, _, _) = amr.brinactivemark(uh, lb, residual, theta=0.4)
            mark = amr.unionmarks(mark, imark)
            mesh = amr.refinemarkedelements(mesh, mark)

        meshHist.append(mesh)

    outfile = "result_sphere_" + amrtype + ".pvd"
    print(f"done ... writing to {outfile} ...")
    V = uh.function_space()
    gap = Function(V, name="gap = u_h - lb").interpolate(uh - lb)
    error = Function(V, name="error = |pi_h(u_exact) - u_h|").interpolate(abs(uexact - uh))
    VTKFile(outfile).write(uh, lb, gap, uexact, error)
    print("")
