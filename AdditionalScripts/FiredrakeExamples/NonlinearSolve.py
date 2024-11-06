import matplotlib.pyplot as plt
from firedrake import *

# Define the mesh and the function space
n = 32  # Number of elements along each axis
mesh = UnitSquareMesh(n, n)

# Define function space
V = FunctionSpace(mesh, "CG", 1)

# Define trial and test functions
u = Function(V)          # Solution
v = TestFunction(V)      # Test function

# Liouville-Bratu parameter
lambda_param = 6.8

# Define the nonlinear form for the Liouville-Bratu equation
F = (dot(grad(u), grad(v)) - lambda_param * exp(u) * v) * dx

# Apply Dirichlet boundary conditions (u = 0 on the boundary)
bc = DirichletBC(V, 0.0, "on_boundary")

# Solve the nonlinear problem using the specified solver parameters
solve(F == 0, u, bcs=bc, solver_parameters={
    'snes_type': 'newtonls',  # Nonlinear solver: Newton with line search
    'ksp_type': 'preonly',    # No Krylov subspace method, just apply the preconditioner
    # Preconditioner: LU factorization (direct solver)
    'pc_type': 'lu'
})

# Output the solution to a file for visualization
File("solution_bratu.pvd").write(u)

# Optionally, visualize the solution using matplotlib
tripcolor(u)
plt.colorbar()
plt.title("Solution to Liouville-Bratu Equation")
plt.show()
