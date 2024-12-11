# Import Firedrake and Netgen
import geopandas as gpd
from firedrake import *
from firedrake.output import VTKFile
try:
    import netgen
except ImportError:
    import sys
    warning("Unable to import NetGen.")
    sys.exit(0)
from netgen.geom2d import SplineGeometry

# Shapely library for computation of Hausdorff distance and Jaccard index
from shapely.geometry import MultiLineString
from shapely import hausdorff_distance


debugoutputs = False
tolerance = 1e-10
width = 2


# Initialize square mesh with netgen
def getInitialMesh(TriHeight):
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-1*width, -1*width),
                     p2=(1*width, 1*width),
                     bc="rectangle")

    ngmsh = geo.GenerateMesh(maxh=TriHeight)
    mesh = Mesh(ngmsh)
    mesh.topology_dm.viewFromOptions('-dm_view')

    return mesh


def getObstacleUfl(mesh, center=[0, 0], width=.9, height=2):
    # Defining the Obstacle
    (x, y) = SpatialCoordinate(mesh)
    dsqr = (x - center[0])**2 + (y - center[1])**2
    psi_ufl = height*exp(-width*dsqr)
    return psi_ufl


# Define ball obstacle problem
def getProblem(mesh):
    # New mesh and old solution unless i == 0
    # Defining FE Space for Solution and Obstacle
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u (FE soln)")

    # Defining Reference Problem (Chapter 12 Bueler (2021))

    # Defining the Obstacle
    psi1 = getObstacleUfl(mesh, [-.85, -.85], 1.5, 5)
    psi2 = getObstacleUfl(mesh, [.85, .85], 1, 5)
    lb = Function(V).interpolate(psi1 + psi2)

    # Set zero Dirichlet Boundary

    gbdry = Function(V).interpolate(Constant(2))
    uexact = gbdry.copy()
    bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
    bcs = DirichletBC(V, gbdry, bdry_ids)

    # weak form problem; F is residual operator in nonlinear system F==0
    v = TestFunction(V)
    # as in Laplace equation:  - div (grad u) = 0
    F = inner(grad(u), grad(v)) * dx

    return (V, u, lb, bcs, F, uexact)

# Solve Obstacle Problem


def getSolver(F, u, lb, bcs, V):
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
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(lb, ub))

    return (F, u, lb, bcs, sp, problem, solver, ub)


def getActiveIndicators(mesh, u, lb):
    W = FunctionSpace(mesh, "DG", 0)
    V = FunctionSpace(mesh, "CG", 1)

    NodalDifference = Function(
        V, name="Nodal Difference").interpolate(abs(u - lb))

    NodalActiveIndicator = Function(V).interpolate(
        conditional(NodalDifference < tolerance, 1, 0))

    ElementActiveIndicator = Function(W).interpolate(
        conditional(NodalDifference < tolerance, 1, 0))

    OuterElementIndicator = Function(W).interpolate(
        conditional(NodalActiveIndicator > 0, conditional(NodalActiveIndicator < 1, 1, 0), 0))

    return ElementActiveIndicator, OuterElementIndicator


def getHausdorff(ComputedPolygon, AnalyticPolygon):

    distance = hausdorff_distance(ComputedPolygon, AnalyticPolygon, .99)

    return distance


def getJaccard(ComputedActiveIndicator, AnalyticActiveIndicator):

    DG0Analytic = AnalyticActiveIndicator._function_space
    AnalyticMesh = AnalyticActiveIndicator._mesh
    ProjComputed = Function(DG0Analytic).project(ComputedActiveIndicator)
    AreaIntersection = assemble(
        ProjComputed * AnalyticActiveIndicator * dx(AnalyticMesh))
    AreaUnion = assemble(
        (ProjComputed + AnalyticActiveIndicator - (ProjComputed * AnalyticActiveIndicator)) * dx(AnalyticMesh))

    # Fixme: Divide by zero check
    return AreaIntersection/AreaUnion


def getGraph(mesh, ElementActiveIndicator, OuterElementIndicator):

    # Get the cell to vertex connectivity
    CellVertexMap = mesh.topology.cell_closure

    BorderElementsIndices = [i for i, value in enumerate(
        OuterElementIndicator.dat.data) if value != 0]
    ActiveSetElementsIndices = [i for i, value in enumerate(
        ElementActiveIndicator.dat.data) if value != 0]

    # Create sets for vertices related to BorderElements and ActiveSet
    BorderVertices = set()
    ActiveVertices = set()

    # Populate BorderVertices set
    for cellIdx in BorderElementsIndices:
        # Add vertices of this border element cell to the set
        # Assuming cells are triangles, adjust if needed
        vertices = CellVertexMap[cellIdx][:3]
        BorderVertices.update(vertices)

    # Populate ActiveVertices set
    for cellIdx in ActiveSetElementsIndices:
        # Add vertices of this active set element cell to the set
        # Assuming cells are triangles, adjust if needed
        vertices = CellVertexMap[cellIdx][:3]
        ActiveVertices.update(vertices)

    # Find intersection of border and active vertices
    FreeBoundaryVertices = BorderVertices.intersection(ActiveVertices)

    # Create an edge set for the FreeBoundaryVertices
    EdgeSet = set()

    # Loop through BorderElements and form edges
    for cellIdx in BorderElementsIndices:
        vertices = CellVertexMap[cellIdx][:3]
        # Check all pairs of vertices in the element
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                v1 = vertices[i]
                v2 = vertices[j]
                # Add edge if both vertices are part of the free boundary
                if v1 in FreeBoundaryVertices and v2 in FreeBoundaryVertices:
                    # Ensure consistent ordering
                    EdgeSet.add((min(v1, v2), max(v1, v2)))

    return FreeBoundaryVertices, EdgeSet


if __name__ == "__main__":

    ComputedTriHeight = .3
    AnalyticTriHeight = .075

    # Generate meshes using netgen
    ComputedMesh = getInitialMesh(ComputedTriHeight)
    AnalyticMesh = getInitialMesh(AnalyticTriHeight)

    ActiveIndicators = []
    LineStringCollections = []

    for mesh in [ComputedMesh, AnalyticMesh]:
        # Define FE space
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V, name="u (FE soln)")
        W = FunctionSpace(mesh, "DG", 0)

        # Define Problem
        (V, u, lb, bcs, F, uexact) = getProblem(mesh)

        # Solve Problem
        (F, u, lb, bcs, sp, problem, solver, ub) = getSolver(F, u, lb, bcs, V)

        #  Construct Element-wise Active Set and Border Active Set indicators
        ElementActiveIndicator, OuterElementIndicator = getActiveIndicators(
            mesh, u, lb)

        # Store ActiveIndicators for Jaccard Computation
        ActiveIndicators.append(ElementActiveIndicator)

        # Constructing
        # Constructs graph for computed free boundary using dmplex indices
        V, E = getGraph(mesh, ElementActiveIndicator, OuterElementIndicator)

        # Convert edges to fd indices
        fdE = [[mesh.topology._vertex_numbering.getOffset(
            edge[0]), mesh.topology._vertex_numbering.getOffset(edge[1])] for edge in list(E)]

        # Map from fd vertex index to coordinates
        coords = mesh.coordinates.dat.data_ro_with_halos

        # Use map fd index edges to coordinate edges
        CoordinateEdges = [[[coords[edge[0]][0], coords[edge[0]][1]], [
            coords[edge[1]][0], coords[edge[1]][1]]] for edge in fdE]

        # construct and store linestring object
        LineStringCollections.append(MultiLineString(CoordinateEdges))

        if debugoutputs:
            VTKFile(f"{mesh}.pvd").write(
                u, lb, ElementActiveIndicator, OuterElementIndicator)

    HausdorffError = getHausdorff(
        LineStringCollections[0], LineStringCollections[1])
    JaccardError = getJaccard(ActiveIndicators[0], ActiveIndicators[1])

    # Jaccard computation using Firedrake.
