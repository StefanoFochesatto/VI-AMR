# from https://www.firedrakeproject.org/demos/poisson_mixed.py.html, then with
# modifications to make it more like what I want

from firedrake import *
mesh = UnitSquareMesh(12, 12)

BDM = FunctionSpace(mesh, "BDM", 1)
DG = FunctionSpace(mesh, "DG", 0)
Z = BDM * DG

sigmau = Function(Z)
sigma, u = split(sigmau)
tau, v = TestFunctions(Z)

x, y = SpatialCoordinate(mesh)
frhs = Function(DG).interpolate(10*exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

F = (dot(sigma, tau) + div(tau) * u + div(sigma) * v)*dx - frhs * v * dx

bc0 = DirichletBC(Z.sub(0), as_vector([0.0, -sin(5*x)]), 3)
bc1 = DirichletBC(Z.sub(0), as_vector([0.0, sin(5*x)]), 4)

solve(F == 0, sigmau, bcs=[bc0, bc1])
sigma, u = sigmau.subfunctions
sigma.rename('sigma')
u.rename('u')

fname = "result_poismix.pvd"
print(f'writing sigma, u to {fname} ...')
VTKFile(fname).write(sigma, u)

if False:
    import matplotlib.pyplot as plt
    from firedrake.pyplot import tripcolor
    fig, axes = plt.subplots()
    colors = tripcolor(u, axes=axes)
    fig.colorbar(colors)
    plt.show()
