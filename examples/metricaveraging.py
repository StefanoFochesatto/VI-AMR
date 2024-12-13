from netgen.geom2d import SplineGeometry
import numpy as np
from firedrake import *
# I found the best way to install animate is to clone to the src folder
# and make install with the firedrake env activated
# git clone https://github.com/mesh-adaptation/animate.git
# cd animate
# make install

from animate import *

from viamr import VIAMR
# Subclass VIAMR to return u


class VIAMRMetric(VIAMR):
    def vcesmark(self, *args, **kwargs):
        # Call the original method
        mark = super().vcesmark(*args, **kwargs)
        return mark, u


try:
    import netgen
except ImportError:
    print("ImportError.  Unable to import NetGen.  Exiting.")
    import sys
    sys.exit(0)


debugoutputs = False

width = 2.0
TriHeight = 0.5

metric_params = {
    "dm_plex_metric": {
        "target_complexity": 3000.0,
        "p": 2.0,  # normalisation order
        "h_min": 1e-07,  # minimum allowed edge length
        "h_max": 1.0,  # maximum allowed edge length
    }
}


def metricfromhessian(mesh, u):
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.set_parameters(metric_params)
    metric.compute_hessian(u)
    metric.normalise()
    return metric


def get_mesh(width, TriHeight):
    # re-generate initial mesh using netgen
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-1 * width, -1 * width),
                     p2=(1 * width, 1 * width), bc="rectangle")
    ngmsh = None
    ngmsh = geo.GenerateMesh(maxh=TriHeight)
    return Mesh(ngmsh)


def sphere_problem(mesh, V):
    # exactly-solvable obstacle problem for spherical obstacle
    # see Chapter 12 of Bueler (2021)
    # obstacle and solution are in function space V
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    psi_ufl = conditional(le(r, r0), sqrt(
        1.0 - r * r), psi0 + dpsi0 * (r - r0))
    psi = Function(V).interpolate(psi_ufl)
    # exact solution is known (and it determines Dirichlet boundary)
    afree = 0.697965148223374
    A = 0.680259411891719
    B = 0.471519893402112
    gbdry_ufl = conditional(le(r, afree), psi_ufl, -A * ln(r) + B)
    gbdry = Function(V).interpolate(gbdry_ufl)
    uexact = gbdry
    return psi, gbdry, uexact


mesh = get_mesh(width, TriHeight)
for i in range(3):
    V = FunctionSpace(mesh, "CG", 1)
    if i == 0:
        u = Function(V)  # set to zero
    else:
        # cross-mesh interpolation
        u = Function(V).interpolate(u)
    u.rename("solution u")

    # Weak form and boundary conditions and obstacle
    lbound, gbdry, uexact = sphere_problem(mesh, V)
    lbound.rename("obstacle psi")
    v = TestFunction(V)
    F = inner(grad(u), grad(v)) * dx
    bdry_ids = (1, 2, 3, 4)  # all four sides of boundary
    bcs = DirichletBC(V, gbdry, bdry_ids)

    # set up nonlinear solver
    sp = {
        "snes_type": "vinewtonrsls",
        "snes_converged_reason": None,
        # "snes_monitor": None,
        # "snes_vi_monitor": None, # prints bounds info for each Newton iteration
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-12,
        "snes_stol": 0.0,
        "snes_vi_zero_tolerance": 1.0e-12,
        "snes_linesearch_type": "basic",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=sp, options_prefix=""
    )
    ubound = Function(V).interpolate(
        Constant(PETSc.INFINITY))  # no upper obstacle
    solver.solve(bounds=(lbound, ubound))

    # Create free boundary indicator
    Vertex, _ = VIAMR().freeboundarygraph(u, lbound, 'fd')
    freeboundaryid = Function(V)
    freeboundaryid.dat.data[list(Vertex)[:]] = 1

    # Build Metrics
    solutionMetric = metricfromhessian(mesh, u)
    freeboundaryMetric = metricfromhessian(mesh, freeboundaryid)

    VImetric = solutionMetric.copy(deepcopy=True)
    VImetric.average(freeboundaryMetric, weights=[.50, .50])
    mesh_intersected = adapt(mesh, VImetric)
    mesh = mesh_intersected

VTKFile("result_MetricMesh.pvd").write(mesh)
