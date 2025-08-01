#!/usr/bin/env python3
"""
Standalone L-shaped domain VI solver with 3 AVM iterations, written by claude.
"""

from firedrake import *
from viamr import VIAMR
import numpy as np

try:
    import netgen
    from netgen.geom2d import SplineGeometry
except ImportError:
    print("Error: NetGen required for L-shaped geometry")
    exit(1)

sp = {
    "snes_type": "vinewtonrsls",
    "snes_rtol": 1e-8,
    "snes_max_it": 200,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

dp = {
    "partition": True,
    "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
}


def create_lshaped_mesh(maxh=0.1):
    """Create L-shaped domain: (-2,5)² \\ [2,5]×[-2,1] (remove 3×3 square in lower right)"""
    geo = SplineGeometry()

    # L-shaped domain: (-2,5)² minus 3×3 square [2,5]×[-2,1] in lower right
    pnts = [
        (2, 1),  # Interior corner point
        (5, 1),  # Top right of removed square
        (5, 5),  # Top right of full domain
        (-2, 5),  # Top left of full domain
        (-2, -2),  # Bottom left of full domain
        (2, -2),  # Bottom left of removed square
    ]
    p1, p2, p3, p4, p5, p6 = [geo.AppendPoint(*pnt) for pnt in pnts]
    curves = [
        [["line", p1, p2], "line"],  # Top of removed square
        [["line", p2, p3], "line"],  # Right side of domain
        [["line", p3, p4], "line"],  # Top of domain
        [["line", p4, p5], "line"],  # Left side of domain
        [["line", p5, p6], "line"],  # Bottom of domain
        [["line", p6, p1], "line"],  # Right side of removed square
    ]
    [geo.Append(c, bc=bc) for c, bc in curves]
    ngmsh = geo.GenerateMesh(maxh=maxh)

    return Mesh(ngmsh, distribution_parameters=dp)


def solve_lshaped_vi():
    """Solve L-shaped VI problem with 3 AVM iterations"""

    # Create initial mesh
    mesh = create_lshaped_mesh(maxh=0.3)
    amr = VIAMR()
    u = None

    # Set up AVM parameters
    amr.setmetricparameters(target_complexity=4000, h_min=1e-3, h_max=1.0)

    for i in range(3):
        print(f"AVM iteration {i+1}/3")

        # Function space
        V = FunctionSpace(mesh, "CG", 1)

        # Initialize or interpolate solution
        if u is None:
            u = Function(V, name="solution")
        else:
            u = Function(V, name="solution").interpolate(u)

        # Define obstacle (upper hemisphere centered at (0,3))
        x, y = SpatialCoordinate(mesh)
        r = sqrt((x - 0) * (x - 0) + (y - 3) * (y - 3))
        r0 = 0.9
        psi0 = np.sqrt(1.0 - r0 * r0)
        dpsi0 = -r0 / psi0
        psi_ufl = conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))

        lb = Function(V, name="obstacle").interpolate(psi_ufl)

        # Boundary conditions (g = -1)
        bc = DirichletBC(V, Constant(-1.0), (1, 2, 3, 4, 5, 6))

        # Solve VI: -∆u = 0, u ≥ ψ, u = -1 on ∂Ω
        v = TestFunction(V)
        F = inner(grad(u), grad(v)) * dx

        problem = NonlinearVariationalProblem(F, u, bc)
        ub = Function(V).interpolate(Constant(float("inf")))

        solver = NonlinearVariationalSolver(problem, solver_parameters=sp)

        solver.solve(bounds=(lb, ub))

        # Apply AVM if not last iteration
        if i < 2:
            mesh = amr.adaptaveragedmetric(mesh, u, lb, gamma=0.2)

    # Compute diagnostics
    DG0 = FunctionSpace(mesh, "DG", 0)
    active_set = Function(DG0, name="active_set")
    active_set.interpolate(conditional(abs(u - lb) < 1e-6, 1.0, 0.0))

    gap = Function(V, name="gap")
    gap.interpolate(u - lb)

    return u, lb, active_set, gap


if __name__ == "__main__":
    u, psi, active_set, gap = solve_lshaped_vi()

    print("Writing result_claudelshaped.pvd")
    VTKFile("result_claudelshaped.pvd").write(u, psi, active_set, gap)

    mesh = u.function_space().mesh()
    print(f"Solved L-shaped VI problem with 3 AVM iterations")
    print(f"Final mesh elements: {mesh.num_cells()}")
    print(f"DOFs: {u.function_space().dim()}")
    print(f"Active set measure: {assemble(active_set * dx):.4f}")
