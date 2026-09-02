import argparse
from firedrake import *
try:
    import netgen
except ImportError:
    import sys
    warning("Unable to import NetGen.")
    sys.exit(0)

from firedrake.petsc import PETSc
from netgen.geom2d import SplineGeometry
import meshio
import numpy as np
import os


tolerance = 1e-10
width = 2


# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description='Generate Sphere Mesh')
parser.add_argument('--iter', type=int, default=5,
                    help='Number of iterations for mesh refinement')
parser.add_argument('--trih', type=float, default=0.45,
                    help='Maximum height of the triangles in the initial mesh')
parser.add_argument('--thresh', type=float, nargs=2,
                    default=[0.2, 0.8], help='Brackets for marking elements')
parser.add_argument('--proc', type=int, default=5,
                    help='Number of processors to use')

args = parser.parse_args()

# Use command-line arguments
max_iterations = args.iter
TriHeight = args.trih
markbracket = args.thresh
proc = args.proc


# Generate initial mesh using netgen
geo = SplineGeometry()
geo.AddRectangle(p1=(-1*width, -1*width),
                 p2=(1*width, 1*width), bc="rectangle")

ngmsh = geo.GenerateMesh(maxh=TriHeight)
mesh = Mesh(ngmsh)
mesh.topology_dm.viewFromOptions('-dm_view')
meshHierarchy = [mesh]
solutionHierarchy = []

# Run with (firedrake) python3 GenerateSphereMesh.py --iter 5 --trih 0.45 --thresh 0.2 0.8 --proc 5


def getActiveInactiveSets(mesh, u):
    # Define psi obstacle function
    W = FunctionSpace(mesh, "DG", 0)
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = - r0 / psi0
    psi_ufl = conditional(le(r, r0), sqrt(1.0 - r * r),
                          psi0 + dpsi0 * (r - r0))
    # Compute pointwise indicator using CG 1
    U = u.function_space()
    psiCG = Function(U).interpolate(psi_ufl)
    DiffCG = Function(U, name="CG Indicator").interpolate(u - psiCG)

    # Using conditional to get elementwise indictor for active set.
    # This piece of code does what we want, all nodes have to be less
    # than the tolerance for element to be active. (Checked by comparing pointwise CG 1 alternative.)
    ActiveSet = Function(W, name="Active Set").interpolate(
        conditional(DiffCG < tolerance, 1, 0))

    # Generating a CG 1 version of the active indicator
    IntersectionCG = Function(U, name="CG Intersection").interpolate(
        conditional(DiffCG < tolerance, 1, 0))

    # Strict conditionals get us elements with non-zero gradient.
    BorderElements = Function(W, name="Border Elements").interpolate(
        conditional(IntersectionCG > 0, conditional(IntersectionCG < 1, 1, 0), 0))

    # Set theory to get inactive set,
    InactiveSet = Function(W, name="Inactive Set").interpolate(
        Constant(-1.0)*((BorderElements + ActiveSet) - Constant(1.0)) + BorderElements)

    return ActiveSet, InactiveSet


def Mark(msh, u, lb, markbracket):
    W = FunctionSpace(msh, "DG", 0)

    # VCES
    # ----------------------------------------------------------------------------------
    V = FunctionSpace(msh, "CG", 1)
    u_ = Function(V, name="CurrentStep")
    CGHeatEQ = Function(V, name="CGHeatEQ")
    v = TestFunction(V)

    DiffCG = Function(V, name="CG Indicator").interpolate(abs(u - lb))
    CGActiveIndicator = Function(V, name="OuterIntersectionCG").interpolate(
        conditional(DiffCG < tolerance, 0, 1))
    u_.assign(CGActiveIndicator)

    timestep = 1.0/int(msh.num_cells()**(.5))

    F = (inner((CGHeatEQ - u_)/timestep, v) +
         inner(grad(CGHeatEQ), grad(v))) * dx
    solve(F == 0, CGHeatEQ)

    DGHeatEQ = Function(W, name="DGHeatEQ").interpolate(CGHeatEQ)

    mark = Function(W, name="Final Marking").interpolate(
        conditional(DGHeatEQ > markbracket[0], conditional(DGHeatEQ < markbracket[1], 1, 0), 0))

    return (mark)


def SolveProblem(mesh, V, u):
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
        problem, solver_parameters=sp, options_prefix="s_")
    # no upper obstacle in fact
    ub = interpolate(Constant(PETSc.INFINITY), V)
    # essentially same as:  solve(F == 0, u, bcs=bcs, ...
    solver.solve(bounds=(lb, ub))

    return u, lb


if __name__ == "__main__":

    # Generate Adaptively Refined Mesh
    for i in range(max_iterations):
        print("level {}".format(i))
        mesh = meshHierarchy[-1]
        # initial iterate is zero
        V = FunctionSpace(mesh, "CG", 1)
        if i == 0:
            u = Function(V, name="u (FE soln)")
        else:
            # Need to define a destination function space to make cross mesh interpolation work
            V_dest = FunctionSpace(mesh, "CG", 1)
            u = interpolate(u, V_dest)

        u, lb = SolveProblem(mesh, V, u)
        solutionHierarchy.append(u)
        (mark) = Mark(mesh, u, lb, markbracket)
        nextmesh = mesh.refine_marked_elements(mark)
        meshHierarchy.append(nextmesh)


# _______________________________________________Distribute AMR_______________________________________________________________

    # Export mesh to Firedrake Compatible File
    meshHierarchy[-2].netgen_mesh.Export("Sphere.msh", "Gmsh2 Format")
    mesh = meshio.read("Sphere.msh")
    mesh.write("Sphere.msh", file_format="gmsh22", binary=False)

    # Run DistributeMesh.py and distribute the mesh
    os.system("mpiexec -n " + str(proc) + " python3 AMRDistributeMesh.py")

    with CheckpointFile("Distributed.h5", 'r') as afile:
        mesh = afile.load_mesh("meshA")
        rank = afile.load_function(mesh, "rank")

    u = solutionHierarchy[-1]
    mesh = meshHierarchy[-2]
    DG0 = FunctionSpace(mesh, "DG", 0)
    rankcheck = Function(DG0).interpolate(rank)

    # Computing Statistics
    ActiveSet, InactiveSet = getActiveInactiveSets(mesh, u)

    ActiveRank = Function(DG0).interpolate(ActiveSet*rankcheck)
    InactiveRank = Function(DG0).interpolate(InactiveSet*rankcheck)

    AMRActiveElementsCount = np.zeros(proc)
    AMRInactiveElementsCount = np.zeros(proc)
    for i in range(1, proc+1):
        AMRActiveRankIndicator = Function(DG0).interpolate(
            conditional(eq(ActiveRank, i), 1, 0))
        AMRInactiveRankIndicator = Function(DG0).interpolate(
            conditional(eq(InactiveRank, i), 1, 0))
        AMRActiveElementsCount[i -
                               1] = np.count_nonzero(AMRActiveRankIndicator.dat.data)
        AMRInactiveElementsCount[i -
                                 1] = np.count_nonzero(AMRInactiveRankIndicator.dat.data)

    AMRmesh = mesh

# _______________________________________________Construct and Distribute Uniform_______________________________________________________________
    # Generate initial mesh using netgen
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-1*width, -1*width),
                     p2=(1*width, 1*width), bc="rectangle")

    ngmsh = geo.GenerateMesh(maxh=TriHeight)
    mesh = Mesh(ngmsh)
    while mesh.num_cells() < meshHierarchy[-3].num_cells():
        Vunif = FunctionSpace(mesh, "DG", 0)
        Unif = Function(Vunif).interpolate(Constant(1))
        Unifmesh = mesh.refine_marked_elements(Unif)
        mesh = Unifmesh
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u (FE soln)")
    u, lb = SolveProblem(mesh, V, u)

   # Generate Distributed Data for Uniform Mesh
    # Export mesh to Firedrake Compatible File
    mesh.netgen_mesh.Export("Unif.msh", "Gmsh2 Format")
    mesh = meshio.read("Unif.msh")
    mesh.write("Unif.msh", file_format="gmsh22", binary=False)

    # Run DistributeMesh.py and distribute the mesh
    os.system("mpiexec -n " + str(proc) + " python3 UnifDistributeMesh.py")

    with CheckpointFile("DistributedUnif.h5", 'r') as afile:
        mesh = afile.load_mesh("meshA")
        rankunif = afile.load_function(mesh, "rank")

    DG0 = FunctionSpace(mesh, "DG", 0)
    rankcheck = Function(DG0).interpolate(rankunif)

    # Computing Statistics
    ActiveSet, InactiveSet = getActiveInactiveSets(mesh, u)

    UnifActiveRank = Function(DG0).interpolate(ActiveSet*rankcheck)
    UnifInactiveRank = Function(DG0).interpolate(InactiveSet*rankcheck)

    UnifActiveElementsCount = np.zeros(proc)
    UnifInactiveElementsCount = np.zeros(proc)
    for i in range(1, proc+1):
        UnifActiveRankIndicator = Function(DG0).interpolate(
            conditional(eq(ActiveRank, i), 1, 0))
        UnifInactiveRankIndicator = Function(DG0).interpolate(
            conditional(eq(InactiveRank, i), 1, 0))
        UnifActiveElementsCount[i -
                                1] = np.count_nonzero(UnifActiveRankIndicator.dat.data)
        UnifInactiveElementsCount[i -
                                  1] = np.count_nonzero(UnifInactiveRankIndicator.dat.data)

    print("Unif Active Elements Count: ", UnifActiveElementsCount)
    print("Unif Inactive Elements Count: ", UnifInactiveElementsCount)
    print("Unif Total Elements: ", mesh.num_cells())
    print("Unif Ratio: ", UnifInactiveElementsCount /
          sum(UnifInactiveElementsCount))

    print("AMR Active Elements Count: ", AMRActiveElementsCount)
    print("AMR Inactive Elements Count: ", AMRInactiveElementsCount)
    print("AMR Total Elements: ", AMRmesh.num_cells())
    print("AMR Ratio: ", AMRInactiveElementsCount /
          sum(AMRInactiveElementsCount))
