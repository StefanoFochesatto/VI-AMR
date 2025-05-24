# This example generates two .pvd files, result_sphere_{udo,vcd}.pvd, suitable
# for a figure in the paper comparing n=1 UDO to default [0.2,0.8] VCD on the
# sphere problem, for which an exact solution is known.

import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
from viamr import VIAMR
from viamr.utility import SphereObstacleProblem

levels = 3  # number of AMR refinements
m0 = 20  # initial mesh is m0 x m0

# miscellaneous
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

for amrtype in ("udo", "vcd"):
    mesh0 = RectangleMesh(m0, m0, Lx=2.0, Ly=2.0, originX=-2.0, originY=-2.0)
    meshHist = [mesh0]
    amr = VIAMR()

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
            u = Function(V, name="u_h")
        else:
            # initialize by cross-mesh interpolation to fine mesh
            uUFL = conditional(u < lb, lb, u)  # use old data
            u = Function(V, name="u_h").interpolate(uUFL)

        v = TestFunction(V)
        F = inner(grad(u), grad(v)) * dx
        bcs = DirichletBC(V, bdryUFL, "on_boundary")
        problem = NonlinearVariationalProblem(F, u, bcs)

        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="s"
        )
        lb = Function(V, name="psi").interpolate(obsUFL)
        ub = Function(V).interpolate(Constant(PETSc.INFINITY))
        solver.solve(bounds=(lb, ub))

        uexact = Function(V, name="u_exact").interpolate(bdryUFL)
        print(f"  ||u - u_exact||_2 = {errornorm(u, uexact):.3e}")

        if i == levels:
            break

        residual = -div(grad(u))
        (imark, _, _) = amr.brinactivemark(u, lb, residual, theta=0.7)
        if amrtype == "udo":
            mark = amr.udomark(u, lb, n=1)
        elif amrtype == "vcd":
            mark = amr.vcdmark(u, lb)
        else:
            raise (ValueError, "unknown amrtype")
        mark = amr.unionmarks(mark, imark)
        mesh = amr.refinemarkedelements(mesh, mark)
        meshHist.append(mesh)

    outfile = "result_sphere_" + amrtype + ".pvd"
    print(f"done ... writing to {outfile} ...")
    V = u.function_space()
    gap = Function(V, name="gap = u - lb").interpolate(u - lb)
    error = Function(V, name="error = |u - u_exact|").interpolate(abs(u - uexact))
    VTKFile(outfile).write(u, lb, gap, uexact, error)
    print("")
