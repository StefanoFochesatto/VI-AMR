from firedrake import *


mesh = RectangleMesh(10, 10, 1, 1, -1, -1)

# Deciding the function space in which we'd like to solve the
# problem. Let's use piecewise linear functions (hence the '1') continuous between
# elements (hence the 'CG')

V = FunctionSpace(mesh, "CG", 1)


v = TestFunction(V)

# Declaring the source function for our problem, as well as the dirichlet boundary
# conditions.
x, y = SpatialCoordinate(mesh)
f = Constant(0.0)
gbdry = Function(V).interpolate((2*(1 + y))/((3 + x)**2 + (1 + y)**2))
bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
bcs = DirichletBC(V, gbdry, bdry_ids)


# Now we define the left and right
# hand sides of our weak from respectively::

u = Function(V)
a = (inner(grad(u), grad(v))) * dx
L = f*v * dx


solve(a == L, u, bcs=[bcs], solver_parameters={
      'ksp_type': 'cg', 'pc_type': 'none'})

# For more details on how to specify solver parameters, see the section
# of the manual on :doc:`solving PDEs <../solving-interface>`.
#
# Next, we might want to look at the result, so we output our solution
# to a file::

File("PoissonReferenceProblem.pvd").write(u)


try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

try:
    from firedrake.pyplot import tripcolor, tricontour
    fig, axes = plt.subplots()
    colors = tripcolor(u, axes=axes)
    fig.colorbar(colors)
except Exception as e:
    warning("Cannot plot figure. Error msg: '%s'" % e)

# The plotting functions in Firedrake mimic those of matplotlib; to produce a
# contour plot instead of a pseudocolor plot, we can call
# :func:`tricontour <firedrake.pyplot.tricontour>` instead::

try:
    fig, axes = plt.subplots()
    contours = tricontour(u, axes=axes)
    fig.colorbar(contours)
except Exception as e:
    warning("Cannot plot figure. Error msg: '%s'" % e)

# Don't forget to show the image::

try:
    plt.show()
except Exception as e:
    warning("Cannot show figure. Error msg: '%s'" % e)

# Alternatively, since we have an analytic solution, we can check the
# :math:`L_2` norm of the error in the solution::

print(sqrt(assemble(dot(u - gbdry, u - gbdry) * dx)))
