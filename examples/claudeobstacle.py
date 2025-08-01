#!/usr/bin/env python3
"""
Standalone Firedrake code for solving a variational inequality (obstacle problem).

Problem: Find u such that
  u >= psi (obstacle constraint)
  <grad(u), grad(v-u)> + <f, v-u> >= 0  for all v >= psi

This implements the obstacle problem on a unit square with Dirichlet boundary conditions.
"""

from firedrake import *
import numpy as np

def solve_obstacle_problem():
    # Create mesh
    n = 32
    mesh = UnitSquareMesh(n, n)
    
    # Function space
    V = FunctionSpace(mesh, "CG", 1)
    
    # Trial and test functions
    u = Function(V)
    v = TestFunction(V)
    
    # Define obstacle function (ensures psi <= 0 on boundary)
    x, y = SpatialCoordinate(mesh)
    psi = Function(V)
    psi.interpolate(0.1 - 2*((x - 0.5)**2 + (y - 0.5)**2))  # Paraboloid obstacle, negative on boundary
    
    # Right-hand side
    f = Function(V)
    f.interpolate(Constant(-10.0))
    
    # Boundary conditions (u = 0 on boundary, compatible with psi <= 0)
    bc = DirichletBC(V, 0.0, "on_boundary")
    
    # Variational inequality solver using SNES
    F = inner(grad(u), grad(v)) * dx - f * v * dx
    
    # Set up the VI bounds (no need to modify bounds since psi <= g = 0 on boundary)
    lb = Function(V)
    lb.assign(psi)  # Lower bound is the obstacle
    ub = Function(V)
    ub.assign(float('inf'))  # No upper bound
    
    # Solve the variational inequality
    problem = NonlinearVariationalProblem(F, u, bc)
    solver = NonlinearVariationalSolver(problem, 
                                       solver_parameters={
                                           'snes_type': 'vinewtonrsls',
                                           'snes_vi_monitor': None,
                                           'ksp_type': 'cg',
                                           'pc_type': 'hypre',
                                           'snes_rtol': 1e-8,
                                           'snes_max_it': 50
                                       })
    
    # Set bounds for VI
    solver.solve(bounds=(lb, ub))
    
    # Compute some diagnostics
    DG0 = FunctionSpace(mesh, "DG", 0)
    active_set = Function(DG0)
    active_set.interpolate(conditional(abs(u - psi) < 1e-6, 1.0, 0.0))
    
    gap = Function(V)
    gap.interpolate(u - psi)
    
    return u, psi, active_set, gap

if __name__ == "__main__":
    u, psi, active_set, gap = solve_obstacle_problem()
    
    # Output all results to one file
    print(f"Writing result_obstacle.pvd")
    u.rename("solution")
    psi.rename("obstacle")
    active_set.rename("active_set")
    gap.rename("gap")
    VTKFile("result_obstacle.pvd").write(u, psi, active_set, gap)
    
    print(f"Solved obstacle problem on 32x32 mesh")
    print(f"DOFs: {u.function_space().dim()}")
    print(f"Active set measure: {assemble(active_set * dx):.4f}")
    print(f"Max solution: {u.dat.data.max():.4f}")
    print(f"Min solution: {u.dat.data.min():.4f}")
