# Import Firedrake and Netgen
from firedrake import *
try:
    import netgen
except ImportError:
    import sys
    warning("Unable to import NetGen.")
    sys.exit(0)
from netgen.geom2d import SplineGeometry

# Shapely library for computation of Hausdorff distance and Jaccard index
from shapely.geometry import Polygon
from shapely import hausdorff_distance

import csv
import math
import argparse


# Set Refinement Scheme Using Kwargs
parser = argparse.ArgumentParser(description='Run Solution')
parser.add_argument('--refinement', type=str, default="AMR",
                    help='type of refinement scheme :AMR, Unif, or Hybrid')
args = parser.parse_args()
Refinement = args.refinement


# Write out each step to paraview for debugging.
debug = 1

# Parameters of experiment
max_iterations = 7
width = 2

if Refinement == "Unif":
    markbracket = [0, 1]
else:
    markbracket = [0.2, 0.8]

TriHeight = .45
tolerance = 1e-10
geo = SplineGeometry()


# Representation of analytic free boundary.
def create_circle(center, radius, num_points):
    # Generate the points on the circle
    points = [(center[0] + radius * math.cos(2 * math.pi * i / num_points),
               center[1] + radius * math.sin(2 * math.pi * i / num_points))
              for i in range(num_points)]

    return Polygon(points)


AnalyticFreeBoundary = create_circle((0, 0), 0.697965148223374, 7500)


# Initialize square mesh with netgen
def InitialMesh():
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-1*width, -1*width),
                     p2=(1*width, 1*width),
                     bc="rectangle")

    ngmsh = geo.GenerateMesh(maxh=TriHeight)
    mesh = Mesh(ngmsh)
    mesh.topology_dm.viewFromOptions('-dm_view')

    return (geo, mesh, ngmsh)


# Define ball obstacle problem
def GetProblem(mesh, u, i):
    # New mesh and old solution unless i == 0
    # Defining FE Space for Solution and Obstacle
    V = FunctionSpace(mesh, "CG", 1)

    # Using cross mesh interpolation for a good initial iterate on each refinement loop
    if i == 0:
        u = Function(V, name="u (FE soln)")
    else:
        V_dest = FunctionSpace(mesh, "CG", 1)
        u = interpolate(u, V_dest)

    # Defining Reference Problem (Chapter 12 Bueler (2021))

    # Defining the Obstacle
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = - r0 / psi0
    psi_ufl = conditional(le(r, r0), sqrt(1.0 - r * r),
                          psi0 + dpsi0 * (r - r0))
    lb = interpolate(psi_ufl, V)

    # Defining the exact solution and Dirichlet Boundary
    afree = 0.697965148223374
    A = 0.680259411891719
    B = 0.471519893402112
    gbdry_ufl = conditional(le(r, afree), psi_ufl, - A * ln(r) + B)
    gbdry = interpolate(gbdry_ufl, V)
    uexact = gbdry.copy()
    bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
    bcs = DirichletBC(V, gbdry, bdry_ids)

    # weak form problem; F is residual operator in nonlinear system F==0
    v = TestFunction(V)
    # as in Laplace equation:  - div (grad u) = 0
    F = inner(grad(u), grad(v)) * dx

    return (V, u, lb, bcs, F, uexact)


# Solve Obstacle Problem
def GetSolver(F, u, lb, bcs, V):
    sp = {"snes_monitor": None,
          "snes_type": "vinewtonrsls",
          "snes_converged_reason": None,
          "snes_rtol": 1.0e-8,
          "snes_atol": 1.0e-12,
          "snes_stol": 1.0e-12,
          "snes_vi_zero_tolerance": 1.0e-12,
          "snes_linesearch_type": "basic",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=sp, options_prefix="")

    # No upper obstacle
    ub = interpolate(Constant(PETSc.INFINITY), V)
    solver.solve(bounds=(lb, ub))

    return (F, u, lb, bcs, sp, problem, solver, ub)


def GetNodalActiveIndicator(msh, u, lb):
    W = FunctionSpace(msh, "DG", 0)
    V = FunctionSpace(msh, "CG", 1)

    NodalDifference = Function(
        V, name="Nodal Difference").interpolate(abs(u - lb))
    NodalActiveIndicator = Function(V).interpolate(
        conditional(NodalDifference < tolerance, 0, 1))
    return NodalActiveIndicator


def Mark(msh, u, lb, iter, Refinement, bracket=[.2, .8]):
    W = FunctionSpace(msh, "DG", 0)
    if bracket == [0, 1]:
        mark = Function(W).interpolate(Constant(1.0))
    else:
        # Variable Coefficient Elliptic Smoothing
        # ----------------------------------------------------------------------------------
        V = FunctionSpace(msh, "CG", 1)
        u0 = Function(V, name="Nodal Active Indicator")
        u1 = Function(V, name="Smoothed Nodal Active Indicator")
        v = TestFunction(V)

        # Compute nodal active set indicator within some tolerance
        NodalDifference = Function(
            V, name="Nodal Difference").interpolate(abs(u - lb))
        NodalActiveIndicator = Function(V).interpolate(
            conditional(NodalDifference < tolerance, 0, 1))

        # Nodal indicator is initial condition to time dependent Heat eq
        u0.assign(NodalActiveIndicator)

        # Vary timestep by average cell circumference of each patch. Applied constant diffusion across all cell sizes
        # (not exactly an average msh.cell_sizes is an L2 projection of the obvious DG0 function into CG1)
        # This forces u and psi to be CG1 unless we work something else out. (like l2 projection into the desired FE space)
        timestep = Function(V)
        timestep.dat.data[:] = 0.5 * msh.cell_sizes.dat.data[:]**2

        # Solve one step implicitly
        F = (inner((u1 - u0)/timestep, v) +
             inner(grad(u1), grad(v))) * dx
        solve(F == 0, u1)

        # Compute average over elements by interpolation into DG0
        u1DG0 = Function(W, name="").interpolate(u1)

        # Applying thresholding parameters
        mark = Function(W, name="Final Marking").interpolate(
            conditional(u1DG0 > bracket[0], conditional(u1DG0 < bracket[1], 1, 0), 0))

    if debug:
        if Refinement == 'AMR':
            towrite = (u, NodalDifference, u0, u1, u1DG0, mark)
            File('debug/AMRMarkingFunction:%s.pvd' % iter).write(*towrite)
        elif Refinement == 'Unif':
            File('debug/UNIFMarkingFunction:%s.pvd' % iter).write(mark)
        else:
            if bracket == [0, 1]:
                File('debug/HybridMarkingFunction:%s(Unif).pvd' %
                     iter).write(mark)
            else:
                towrite = (u, NodalDifference, u0, u1, u1DG0, mark)
                File('debug/HybridMarkingFunction:%s(AMR).pvd' %
                     iter).write(*towrite)

    return mark


# Utility Functions for Computing Jaccard and Hausdorff error.

def GetComputedFreeBoundary(mesh, NodalActiveIndicator):
    # Define psi obstacle function
    W = FunctionSpace(mesh, "DG", 0)
    OuterElements = Function(W, name="OuterNeighborElements").interpolate(conditional(
        NodalActiveIndicator > 0, conditional(NodalActiveIndicator < 1, 1, 0), 0))

    # Search for active vertices (slow)
    (ActiveVertices, ActiveVerticesIdx) = GetActiveVertices(
        NodalActiveIndicator, OuterElements, mesh)

    # Sort vertices so that we can form a polygon
    sorted = SortPointsClockwise(set(ActiveVertices))
    ComputedFreeBoundary = Polygon(sorted)

    # DebugCode
    # FreeboundaryIndicator = Function(U, name="FreeBoundaryIndicator")
    # FreeboundaryIndicator.dat.data[ActiveVerticesIdx] = 1
    # towrite = (DiffCG, ActiveSet, ActiveSetCG,
    #           OuterElements, FreeboundaryIndicator)
    # File('AreaMetric/Test: %s.pvd' % iter).write(*towrite)

    return ComputedFreeBoundary


def SortPointsClockwise(points):
    # Compute the centroid of the points
    centroid = tuple(map(lambda x: sum(x) / len(points), zip(*points)))

    # Function to compute the angle relative to the centroid
    def AngleFromCentroid(point):
        dx = point[0] - centroid[0]
        dy = point[1] - centroid[1]
        return math.atan2(dy, dx)

    # Sort the points based on the angle in descending order (clockwise)
    SortedPoints = sorted(points, key=AngleFromCentroid, reverse=True)
    # Append the first point to the end of the list to form a closed loop
    SortedPoints.append(SortedPoints[0])

    return SortedPoints


def GetActiveVertices(NodalActiveIndicator, OuterElements, mesh):
    # Construct cell to vertex map
    coords = mesh.coordinates.dat.data_ro_with_halos
    cell_to_vertices = mesh.coordinates.cell_node_map().values_with_halo

    # Init lists
    ActiveVertices = []
    ActiveVerticesIdx = []

    # Iterate over the cells' closure
    for CellIdx in range(mesh.topology.num_cells()):
        # Check if the current cell is marked by OuterElements
        if OuterElements.dat.data[CellIdx] == 1:
            CellClosure = cell_to_vertices[CellIdx]
            for NodeIdx in CellClosure:
                # Check if this vertex has a value of 1 in NodalActiveIndicator
                if NodalActiveIndicator.at(coords[NodeIdx]) >= .5:
                    ActiveVertices.append(tuple(coords[NodeIdx]))
                    ActiveVerticesIdx.append(NodeIdx)

    return ActiveVertices, ActiveVerticesIdx


def GetJaccard(polygon1, polygon2):
    # Ensure the inputs are Shapely Polygons
    if not isinstance(polygon1, Polygon) or not isinstance(polygon2, Polygon):
        raise ValueError("Both inputs must be Shapely Polygon objects.")

    # Compute the intersection and union
    intersection = polygon1.intersection(polygon2)
    union = polygon1.union(polygon2)

    # Calculate the Jaccard
    intersection_area = intersection.area
    union_area = union.area
    iou = intersection_area / union_area
    return iou


def GetHausdorff(ComputedFreeBoundary, AnalyticFreeBoundary):
    # Compute the Hausdorff distance
    distance = hausdorff_distance(
        ComputedFreeBoundary, AnalyticFreeBoundary, .99)
    print("Hausdorff Distance: ", distance)
    return distance


# Run AMR Loop
def RunSolution(max_iterations):
    # Initialize lists for error metrics
    l2 = []
    IoU = []
    Hausdorff = []
    # Count for hybrid strategy
    count = []
    h = TriHeight

    # Generate initial mesh using netgen
    (geo, mesh, ngmsh) = InitialMesh()
    meshHierarchy = [mesh]

    # Define FE space
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u (FE soln)")

    for i in range(max_iterations):
        print("level {}: {}".format(i, Refinement))

        mesh = meshHierarchy[-1]

        # Define Problem
        (V, u, lb, bcs, F, uexact) = GetProblem(mesh, u, i)

        # Solve Problem
        (F, u, lb, bcs, sp, problem, solver, ub) = GetSolver(F, u, lb, bcs, V)

        # Get Nodal Active Indicator
        NodalActiveIndicator = GetNodalActiveIndicator(mesh, u, lb)

        # Compute Shapely Polygon for Computed Free Boundary
        ComputedFreeBoundary = GetComputedFreeBoundary(
            mesh, NodalActiveIndicator)

        # Compute Jaccard index
        JaccardError = GetJaccard(ComputedFreeBoundary, AnalyticFreeBoundary)
        IoU.append(JaccardError)

        # Compute Hausdorff error
        HausdorffError = GetHausdorff(
            ComputedFreeBoundary, AnalyticFreeBoundary)
        Hausdorff.append(HausdorffError)

        # Compute L2 Error
        diffu = interpolate(u - uexact, V)
        L2Error = sqrt(assemble(dot(diffu, diffu) * dx))
        l2.append(L2Error)

        # Hybrid Refinement Strategy:
        # 1. Compute ratio between Hausdorff Error and h^2
        # 2. If Hausdorff Error < h^2 or ratio is within .1 of 1:
        #       apply uniform
        #    Else:
        #        apply AMR

        if Refinement == "Hybrid":
            ratio = HausdorffError/(h**2)
            switch = math.isclose(ratio, 1, rel_tol=.1)

            if HausdorffError < h**2 or switch:
                mark = Mark(mesh, u, lb, i, Refinement, bracket=[0, 1])
                nextmesh = mesh.refine_marked_elements(mark)
                meshHierarchy.append(nextmesh)
                h = h*(1/2)
                count.append(0)

            else:
                mark = Mark(mesh, u, lb, i, Refinement, markbracket)
                nextmesh = mesh.refine_marked_elements(mark)
                meshHierarchy.append(nextmesh)
                count.append(1)
        else:
            mark = Mark(mesh, u, lb, i, Refinement, markbracket)
            nextmesh = mesh.refine_marked_elements(mark)
            meshHierarchy.append(nextmesh)

    return (l2, IoU, Hausdorff, meshHierarchy)

# Mesh Details for Convergence Plots


def GetMeshDetails(mesh, name='', color=None):
    '''Print mesh information using DMPlex numbering.'''
    plex = mesh.topology_dm             # DMPlex
    coords = plex.getCoordinatesLocal()  # Vec
    vertices = plex.getDepthStratum(0)  # pair
    edges = plex.getDepthStratum(1)     # pair
    triangles = plex.getDepthStratum(2)  # pair
    ntriangles = triangles[1]-triangles[0]
    nvertices = vertices[1]-vertices[0]
    nedges = edges[1]-edges[0]
    print(color % '%s has %d elements, %d vertices, and %d edges:' %
          (name, ntriangles, nvertices, nedges))
    return (ntriangles, nvertices, nedges)


# Run solution strategy and collate data to csv file for plotting
if __name__ == "__main__":

    (l2, IoU, Hausdorff, meshHierarchy) = RunSolution(max_iterations)

    Refinements = list(range(len(meshHierarchy) - 1))
    Elements = []
    dof = []
    for mesh in meshHierarchy:
        (ntriangles, nvertices, nedges) = GetMeshDetails(
            mesh, name=f'{Refinement} Mesh', color=BLUE)
        Elements.append(ntriangles)
        dof.append(nvertices)
    Elements.pop()
    dof.pop()

    OutputFile = f"{Refinement}.csv"

    # Write the arrays to the CSV file
    with open(OutputFile, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['L2', 'IoU', 'Hausdorff', 'Elements', 'dof'])
        for data in zip(l2, IoU, Hausdorff, Elements, dof):
            writer.writerow(data)  # Write each row of data
