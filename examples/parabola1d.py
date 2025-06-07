import numpy as np
import matplotlib.pyplot as plt
from firedrake import *

MeshSizes = [10, 20, 40, 100, 200, 400, 1000]

h = [2.0 / i for i in MeshSizes]
errors = {"gap": [], "H1": [], "L2": []}

# exact free boundaries
b = (2 - (2) ** 0.5) * 0.5
a = -b

for i in MeshSizes:
    mesh = IntervalMesh(i, -1, 1)
    V = FunctionSpace(mesh, "CG", 1)

    x = SpatialCoordinate(mesh)[0]
    psi_ufl = 0.5 - x ** 2
    lb = Function(V).interpolate(psi_ufl)
    bcs = DirichletBC(V, Constant(0.0), (1, 2))

    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v)) * dx

    sp = {  # "snes_monitor": None,
        "snes_type": "vinewtonrsls",
        "snes_converged_reason": None,
        "snes_max_it": 1000,
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-12,
        "snes_stol": 1.0e-12,
        "snes_vi_zero_tolerance": 1.0e-12,
        "snes_linesearch_type": "basic",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=sp, options_prefix="s"
    )
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(lb, ub))

    leftline = ((0.5 - a ** 2) / (a + 1)) * (x + 1)
    rightline = -((0.5 - b ** 2) / (1 - b)) * (x - 1)
    u_exact_ufl = conditional(
        le(x, a), leftline, conditional(ge(x, b), rightline, psi_ufl)
    )
    CG3 = FunctionSpace(mesh, "CG", 3)
    uexact = Function(CG3).interpolate(u_exact_ufl)

    gap = np.min(np.abs(mesh.topology_dm.getCoordinates().array - b))
    errors["gap"].append(gap)
    errors["H1"].append(errornorm(uexact, u, norm_type="H1"))
    errors["L2"].append(errornorm(uexact, u, norm_type="L2"))

h1label = "$=\|u-u_h\|_{H^1}$"
l2label = "$=\|u-u_h\|_{L^2}$"
plt.loglog(h, errors["H1"], "o", ms=10.0, color="k", mfc="w", label=h1label)
plt.loglog(h, errors["L2"], "s", ms=10.0, color="k", mfc="w", label=l2label)
plt.loglog(h, errors["gap"], "^", ms=10.0, color="r", mfc="w", label="$=\Delta_h$")

# p_H1 = np.polyfit(np.log(h), np.log(errors["H1"]), 1)
# h1label = "$=\|u-u_h\|_{H^1} = O(h^{%.2f})$" % p_H1[0]
# p_L2 = np.polyfit(np.log(h), np.log(errors["L2"]), 1)
# l2label = "$=\|u-u_h\|_{L^2} = O(h^{%.2f})$" % p_L2[0]
# plt.loglog(h, np.exp(p_H1[0] * np.log(h) + p_H1[1]), "--", lw=0.5, color="k")
# plt.loglog(h, np.exp(p_L2[0] * np.log(h) + p_L2[1]), "--", lw=0.5, color="k")

plt.axis([0.001, 0.35, 2.5e-7, 1.0])
plt.xlabel("h", fontsize=12.0)
plt.ylabel("error", fontsize=12.0)
plt.legend(fontsize=14.0, loc="upper left", labelcolor="linecolor")

outname = "parabola1d.png"
print(f"saving figure in {outname} ...")
plt.savefig(outname, bbox_inches="tight")
