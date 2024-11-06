from firedrake import *
import matplotlib.pyplot as plt
import numpy as np


MeshSizes = [10, 20, 40, 100, 200, 400, 1000]  # the worst
ExactMeshSize = 4096


# Initializing lists to store results
HistError_LInf = []
HistError_L2 = []
HistError_H1 = []
h = [2.0 / i for i in MeshSizes]


# Problems and Exact Solution
# u'' = -1  u(1) = 0 u(-1) = 0
# So u' = -x + C1
# So u = -x^2/2 + C1x + C2
# u(1) = 0 = -1/2 + C1 + C2
# u(-1) = 0 = -1/2 - C1 + C2
# C1 = 0, C2 = 1/2
# So u = 1/2 - x^2/2

# Generating UFL for Exact Solution
ExactMesh = IntervalMesh(ExactMeshSize, -1, 1)
VExact = FunctionSpace(ExactMesh, 'CG', 1)
xx = SpatialCoordinate(ExactMesh)[0]
u_exact_ufl = (1/2) - (xx**2)/2


for i in MeshSizes:
    # Initialize Mesh
    mesh = IntervalMesh(i, -1, 1)

    # Initialize Function Space
    V = FunctionSpace(mesh, 'CG', 1)

    # Define Dirichlet Boundary Conditions and Obstacle
    x = SpatialCoordinate(mesh)[0]  # <- use [0] in one-dimensional case
    gbdry = Function(V).interpolate(Constant(0.0))
    bdry_ids = (1, 2)
    bcs = DirichletBC(V, gbdry, bdry_ids)

    # Define Weak Form
    f = Function(V).interpolate(Constant(1.0))
    u = TrialFunction(V)
    v = TestFunction(V)

    a = (inner(grad(u), grad(v))) * dx
    L = f*v*dx

    uu = Function(V)

    solve(a == L, uu, bcs=bcs, solver_parameters={"ksp_type": "preonly",
                                                  "pc_type": "lu"})

    uuExact = Function(VExact).interpolate(uu)
    uerr = Function(VExact).interpolate(u_exact_ufl - uuExact)

    # Computing Error Norms
    error_LInf = np.max(np.abs(uerr.dat.data))
    error_L2 = norm(uerr)
    error_H1 = norm(uerr, 'H1')

    HistError_LInf.append(error_LInf)
    HistError_L2.append(error_L2)
    HistError_H1.append(error_H1)


p_L2 = np.polyfit(np.log(h), np.log(HistError_L2), 1)
p_H1 = np.polyfit(np.log(h), np.log(HistError_H1), 1)

print('Convergence Rate L2 = ', p_L2[0])
print('Convergence Rate H1 = ', p_H1[0])


plt.loglog(h, HistError_L2, 'o', color='C2',
           label=f'L2 Error O(h^{p_L2[0]:.2f})')
plt.loglog(h, np.exp(p_L2[0] * np.log(h) + p_L2[1]), '--', color='C2')
plt.loglog(h, HistError_H1, 'o', color='C3',
           label=f'H1 Error O(h^{p_H1[0]:.2f})')
plt.loglog(h, np.exp(p_H1[0] * np.log(h) + p_H1[1]), '--', color='C3')
plt.xlabel('h')
plt.gca().set_xlim([0.001, 1.0])
plt.ylabel('Error')
plt.legend()
plt.show()
