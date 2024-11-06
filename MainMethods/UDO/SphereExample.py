from collections import deque
from firedrake import *
try:
    import netgen
except ImportError:
    import sys
    warning("Unable to import NetGen.")
    sys.exit(0)

from firedrake.petsc import PETSc
from netgen.geom2d import SplineGeometry

import numpy as np


tolerance = 1e-10
max_iterations = 8
n = 3


# Generate initial mesh using netgen
width = 2
TriHeight = .3
geo = SplineGeometry()
geo.AddRectangle(p1=(-1*width, -1*width),
                 p2=(1*width, 1*width),
                 bc="rectangle")

ngmsh = geo.GenerateMesh(maxh=TriHeight)
mesh = Mesh(ngmsh)
mesh.topology_dm.viewFromOptions('-dm_view')
meshHierarchy = [mesh]


# Fast Multi Neighbor Lookup BFS can Avoid Active Set
def mark_neighbors(mesh, func, func_space, levels, ActiveSet):
    # Create a new DG0 function to store the result
    result = Function(func_space, name='nNeighbors')

    # Create a dictionary to map each vertex to the cells that contain it
    vertex_to_cells = {}

    # Get the cell to vertex connectivity
    cell_vertex_map = mesh.topology.cell_closure

    # Loop over all cells to populate the dictionary
    for i in range(mesh.num_cells()):
        # Only consider the first three entries, which correspond to the vertices
        for vertex in cell_vertex_map[i][:3]:
            if vertex not in vertex_to_cells:
                vertex_to_cells[vertex] = []
            vertex_to_cells[vertex].append(i)

    # Loop over all cells
    for i in range(mesh.num_cells()):
        # If the function value is 1 and the cell is in the active set
        if func.dat.data[i] == 1 and ActiveSet.dat.data[i] == 0:
            # Use a BFS algorithm to find all cells within the specified number of levels
            queue = deque([(i, 0)])
            visited = set()
            while queue:
                cell, level = queue.popleft()
                if cell not in visited and level <= levels:
                    visited.add(cell)
                    result.dat.data[cell] = 1
                    for vertex in cell_vertex_map[cell][:3]:
                        for neighbor in vertex_to_cells[vertex]:
                            # if ActiveSet.dat.data[neighbor] == 0:
                            queue.append((neighbor, level + 1))

    return result


def NodeMark(msh, u, lb, iter, n=3):
    # Unstructured Dilation Operator
    # ----------------------------------------------------------------------------------

    # Compute pointwise indicator using CG 1
    U = FunctionSpace(msh, "CG", 1)
    W = FunctionSpace(msh, "DG", 0)

    NodalDifference = Function(U, name="Difference").interpolate(abs(u - lb))

    # Using conditional to get element-wise indictor for active set
    ElementActiveSetIndicator = Function(W, name="ElementActiveSetIndicator").interpolate(
        conditional(NodalDifference < tolerance, 1, 0))

    NodalActiveSetIndicator = Function(U, name="NodalActiveSetIndicator").interpolate(
        conditional(NodalDifference < tolerance, 1, 0))

    # Define Border Elements Set
    BorderElements = Function(W, name="BorderElements").interpolate(conditional(
        NodalActiveSetIndicator > 0, conditional(NodalActiveSetIndicator < 1, 1, 0), 0))

    # mark_neighbors constructs N^n(B) indicator. Argument ElementActiveSetIndicator is for
    # option to refine only in active or only in inactive set (currently commented out).
    mark = mark_neighbors(msh, BorderElements, W, n, ElementActiveSetIndicator)

    if debug:
        towrite = (u, NodalDifference, ElementActiveSetIndicator,
                   NodalActiveSetIndicator, BorderElements, mark)
        File(
            'DebugSphere/UDO[{}]/iterate:{}.pvd'.format(n, iter)).write(*towrite)

    return mark


if __name__ == "__main__":
    for i in range(max_iterations):

        print("level {}".format(i))
        mesh = meshHierarchy[-1]
        # obstacle and solution are in P1
        V = FunctionSpace(mesh, "CG", 1)
        (x, y) = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        # see Chapter 12 of Bueler (2021)
        r0 = 0.9
        psi0 = np.sqrt(1.0 - r0 * r0)
        dpsi0 = - r0 / psi0
        psi_ufl = conditional(le(r, r0), sqrt(1.0 - r * r),
                              psi0 + dpsi0 * (r - r0))
        lb = interpolate(psi_ufl, V)
        # exact solution is known (and it determines Dirichlet boundary)
        afree = 0.697965148223374
        A = 0.680259411891719
        B = 0.471519893402112
        gbdry_ufl = conditional(le(r, afree), psi_ufl, - A * ln(r) + B)
        gbdry = interpolate(gbdry_ufl, V)
        uexact = gbdry.copy()

        # initial iterate is zero
        if i == 0:
            u = Function(V, name="u (FE soln)")
        else:
            # Need to define a destination function space to make cross mesh interpolation work
            V_dest = FunctionSpace(mesh, "CG", 1)
            u = interpolate(u, V_dest)

        # weak form problem; F is residual operator in nonlinear system F==0
        v = TestFunction(V)
        # as in Laplace equation:  - div (grad u) = 0
        F = inner(grad(u), grad(v)) * dx
        bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
        bcs = DirichletBC(V, gbdry, bdry_ids)

        # problem is nonlinear so we need a nonlinear solver, from PETSc's SNES component
        # specific solver is a VI-adapted line search Newton method called "vinewtonrsls"
        # see reference:
        #   S. Benson and T. Munson (2006). Flexible complementarity solvers for large-scale applications,
        #       Optimization Methods and Software, 21, 155–168.
        sp = {"snes_vi_monitor": None,         # prints residual norms for each Newton iteration
              "snes_type": "vinewtonrsls",
              "snes_converged_reason": None,  # prints CONVERGED_... message at end of solve
              "snes_rtol": 1.0e-8,
              "snes_atol": 1.0e-12,
              "snes_stol": 1.0e-12,
              "snes_vi_zero_tolerance": 1.0e-12,
              "snes_linesearch_type": "basic",
              # these 3 options say Newton step equations are solved by LU
              "ksp_type": "preonly",
              "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps"}
        problem = NonlinearVariationalProblem(F, u, bcs)
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="")
        ub = interpolate(Constant(PETSc.INFINITY), V)
        solver.solve(bounds=(lb, ub))

        (mark) = NodeMark(mesh, u, lb, i, n)

        nextmesh = mesh.refine_marked_elements(mark)
        meshHierarchy.append(nextmesh)
