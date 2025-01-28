import numpy as np
from collections import deque

from firedrake import *
from firedrake.petsc import OptionsManager, PETSc
from firedrake.output import VTKFile
from firedrake.utils import IntType
import firedrake.cython.dmcommon as dmcommon

from pyop2.mpi import MPI

from shapely.geometry import MultiLineString
import shapely


class VIAMR(OptionsManager):
    def __init__(self, **kwargs):
        self.activetol = kwargs.pop("activetol", 1.0e-10)
        # if True, add (slow) extra checking
        self.debug = kwargs.pop("debug", False)
        self.metricparameters = None

    def spaces(self, mesh, p=1):
        """Return CG{p} and DG{p-1} spaces."""
        assert isinstance(p, int)
        assert p >= 1
        return FunctionSpace(mesh, "CG", p), FunctionSpace(mesh, "DG", p - 1)

    def meshsizes(self, mesh):
        """Compute number of vertices, number of elements, and range of
        mesh diameters.  Valid in parallel."""
        CG1, DG0 = self.spaces(mesh, p=1)
        nvertices = CG1.dim()
        nelements = DG0.dim()
        hmin = float(mesh.comm.allreduce(min(mesh.cell_sizes.dat.data_ro), op=MPI.MIN))
        hmax = float(mesh.comm.allreduce(max(mesh.cell_sizes.dat.data_ro), op=MPI.MAX))
        return nvertices, nelements, hmin, hmax

    def nodalactive(self, u, lb):
        """Compute nodal active set indicator in same function space as u.
        Applies to unilateral obstacle problems with u >= lb.  The active
        set is {x : u(x) == lb(x)}, within activetol."""
        if self.debug:
            assert min(u.dat.data_ro - lb.dat.data_ro) >= 0.0
        z = Function(u.function_space(), name="Nodal Active Set Indicator")
        z.interpolate(conditional(abs(u - lb) < self.activetol, 0, 1))
        return z

    def elemactive(self, u, lb):
        """Compute element active set indicator in DG0."""
        if self.debug:
            assert min(u.dat.data_ro - lb.dat.data_ro) >= 0.0
        _, DG0 = self.spaces(u.function_space().mesh())
        z = Function(DG0, name="Element Active Set Indicator")
        z.interpolate(conditional(abs(u - lb) < self.activetol, 1, 0))
        return z

    def elemborder(self, nodalactive):
        """From nodal activeset indicator compute bordering element indicator."""
        if self.debug:
            assert min(nodalactive.dat.data_ro) >= 0.0
            assert max(nodalactive.dat.data_ro) <= 1.0
        _, DG0 = self.spaces(nodalactive.function_space().mesh())
        z = Function(DG0, name="Border Elements")
        z.interpolate(
            conditional(nodalactive > 0, conditional(nodalactive < 1, 1, 0), 0)
        )
        return z

    def bfs_neighbors(self, mesh, border, levels, active):
        """element-wise Fast Multi Neighbor Lookup BFS can Avoid Active Set"""

        # dictionary to map each vertex to the cells that contain it
        vertex_to_cells = {}
        cell_vertex_map = mesh.topology.cell_closure  # cell to vertex connectivity
        # Loop over all cells to populate the dictionary
        for i in range(mesh.num_cells()):
            # first three entries correspond to the vertices
            for vertex in cell_vertex_map[i][:3]:
                if vertex not in vertex_to_cells:
                    vertex_to_cells[vertex] = []
                vertex_to_cells[vertex].append(i)

        # Loop over all cells to mark neighbors
        # Create a new DG0 function to store the result
        result = Function(border.function_space(), name="nNeighbors")
        for i in range(mesh.num_cells()):
            # If the function value is 1 and the cell is in the active set
            if border.dat.data[i] == 1 and active.dat.data[i] == 0:
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
                                queue.append((neighbor, level + 1))
        return result

    def udomark(self, mesh, u, lb, n=2):
        """Mark mesh using Unstructured Dilation Operator (UDO) algorithm.
        Warning: Not valid in parallel."""

        # generate element-wise and nodal-wise indicators for active set
        _, DG0 = self.spaces(mesh)
        nodalactive = self.nodalactive(u, lb)
        elemactive = self.elemactive(u, lb)

        # generate border element indicator
        elemborder = self.elemborder(nodalactive)

        # bfs_neighbors() constructs N^n(B) indicator.  Last argument
        # is to refine only in active or only in inactive set (currently commented out).
        return self.bfs_neighbors(mesh, elemborder, n, elemactive)

    def vcesmark(self, mesh, u, lb, bracket=[0.2, 0.8], returnSmooth=False):
        """Mark mesh using Variable Coefficient Elliptic Smoothing (VCES) algorithm.
        Valid in parallel."""

        # Compute nodal active set indicator within some tolerance
        CG1, DG0 = self.spaces(mesh)
        nodalactive = self.nodalactive(u, lb)

        # Vary timestep by average cell area of each patch.
        # Not exactly an average because msh.cell_sizes is an L2 projection of
        # the obvious DG0 function into CG1.
        timestep = Function(CG1)
        timestep.dat.data[:] = 0.5 * mesh.cell_sizes.dat.data[:] ** 2

        # Solve one step implicitly using a linear solver
        # Nodal indicator is initial condition to time dependent Heat eq
        w = TrialFunction(CG1)
        v = TestFunction(CG1)
        a = w * v * dx + timestep * inner(grad(w), grad(v)) * dx
        L = nodalactive * v * dx
        u = Function(CG1, name="Smoothed Nodal Active Indicator")
        solve(a == L, u)

        if returnSmooth:
            return u

        else:
            # Compute average over elements by interpolation into DG0
            DG0 = FunctionSpace(mesh, "DG", 0)
            uDG0 = Function(DG0, name="Smoothed Nodal Active Indicator as DG0")
            uDG0.interpolate(u)

            # Applying thresholding parameters
            mark = Function(DG0, name="VCES Marking")
            mark.interpolate(
                conditional(uDG0 > bracket[0], conditional(uDG0 < bracket[1], 1, 0), 0)
            )

            return mark

    def setmetricparameters(self, **kwargs):
        self.target_complexity = kwargs.pop("target_complexity", 3000.0)
        self.h_min = kwargs.pop("h_min", 1.0e-7)
        self.h_max = kwargs.pop("h_max", 1.0)
        mp = {
            "target_complexity": self.target_complexity,  # target number of nodes
            "p": 2.0,  # normalisation order
            "h_min": self.h_min,  # minimum allowed edge length
            "h_max": self.h_max,  # maximum allowed edge length
        }
        self.metricparameters = {"dm_plex_metric": mp}
        return None

    def metricfromhessian(self, mesh, u):
        """Construct a hessian based metric from a solution"""

        assert (
            self.metricparameters is not None
        ), "call setmetricparameters() before calling metricfromhessian()"

        from animate import RiemannianMetric

        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)
        metric.set_parameters(self.metricparameters)
        metric.compute_hessian(u)
        metric.normalise()
        return metric

    def metricrefine(self, mesh, u, lb, weights=[0.50, 0.50]):
        """Implementation of anisotropic metric based refinement which is free boundary aware. Constructs both the
        hessian based metric and an isotropic metric based off of the magnitude of the gradient of the smoothed vces indicator.  These metrics are averaged using the weights."""

        assert (
            self.metricparameters is not None
        ), "call setmetricparameters() before calling metricrefine()"

        from animate import adapt, RiemannianMetric

        # Construct isotropic metric from abs(grad(smoothed_vces_indicator))
        V, _ = self.spaces(mesh)
        dim = mesh.topological_dimension()

        # Get magnitude of gradients
        s = self.vcesmark(mesh, u, lb, returnSmooth=True)
        ags = Function(V).interpolate(sqrt(dot(grad(s), grad(s))))

        # Constructing metric. Basically "L2" option in metric.compute_isotropic_metric,
        # however we already have a P1 indicator
        freeboundaryMetric = RiemannianMetric(TensorFunctionSpace(mesh, "CG", 1))
        freeboundaryMetric.set_parameters(self.metricparameters)
        freeboundaryMetric.interpolate(ags * ufl.Identity(dim))
        freeboundaryMetric.normalise()

        # Build hessian based metric for interpolation error and average
        solutionMetric = self.metricfromhessian(mesh, u)
        VImetric = solutionMetric.copy(deepcopy=True)
        VImetric.average(freeboundaryMetric, weights=weights)

        # Adapt
        adaptedMesh = adapt(mesh, VImetric)
        return adaptedMesh
    
    def refinemarkedelements(self, mesh, indicator):
        """petsc4py implementation of .refine_marked_elements(), works in parallel only tested in 2D. Still working out the kinks on more than one iteration of refinement."""
        
        
        # Create Section for DG0 indicator
        tdim = mesh.topological_dimension()
        entity_dofs = np.zeros(tdim+1, dtype=IntType)
        entity_dofs[:] = 0
        entity_dofs[-1] = 1
        indicatorSect, _ = dmcommon.create_section(mesh, entity_dofs)

        # Pull Plex from mesh
        dm = mesh.topology_dm
        
        # Create an adaptation label to mark cells for refinement
        dm.createLabel('refine')
        adaptLabel = dm.getLabel('refine')
        adaptLabel.setDefaultValue(0)

        # dmcommon provides a python binding for this operation of setting the label given an indicator function data array
        dmcommon.mark_points_with_function_array(
            dm, indicatorSect, 0, indicator.dat.data_with_halos, adaptLabel, 1)

        # Create a DMPlexTransform object to apply the refinement
        opts = PETSc.Options()
        opts['dm_plex_transform_active'] = 'refine'
        opts['dm_plex_transform_type'] = 'refine_sbr' # <- skeleton based refinement is what netgen does.
        dmTransform = PETSc.DMPlexTransform().create()
        dmTransform.setDM(dm)
        # For now the only way to set the active label with petsc4py is with PETSc.Options() (DMPlexTransformSetActive() has no binding)
        dmTransform.setFromOptions()
        dmTransform.setUp()
        dmAdapt = dmTransform.apply(dm)
        
        # Labels are no longer needed, not sure if we need to call destroy on them. 
        dmAdapt.removeLabel('refine')
        dm.removeLabel('refine')
        dmTransform.destroy()

        # Pull distribution parameters from original dm
        distParams = indicator.function_space().mesh()._distribution_parameters
        
        # Create a new mesh from the adapted dm
        refinedmesh = Mesh(dmAdapt, distribution_parameters=distParams)
        
        return refinedmesh

    
    

    def jaccard(self, active1, active2):
        """Compute the Jaccard metric from two element-wise active
        set indicators.  These indicators must be DG0 functions, but they
        can be on different meshes.  By definition, the Jaccard metric of
        two sets is
            J(S,T) = |S cap T| / |S cup T|,
        that is, the ratio of the area (measure) of the intersection
        divided by the area of the union.
        Warning: Not valid in parallel."""
        # FIXME how to check that active1, active2 are in DG0 spaces?
        # FIXME fails in parallel; the line generating proj2 will throw
        #     AssertionError: Whoever made mesh_B should explicitly mark
        #     mesh_A as having a compatible parallel layout.
        assert (
            active1.function_space().mesh()._comm.size == 1
        ), "jaccard() not valid in parallel"
        if self.debug:
            for a in [active1, active2]:
                assert min(a.dat.data_ro) >= 0.0
                assert max(a.dat.data_ro) <= 1.0
        mesh1 = active1.function_space().mesh()
        proj2 = Function(active1.function_space()).project(active2)
        AreaIntersection = assemble(proj2 * active1 * dx(mesh1))
        AreaUnion = assemble((proj2 + active1 - (proj2 * active1)) * dx(mesh1))
        assert AreaUnion > 0.0
        return AreaIntersection / AreaUnion

    def hausdorff(self, E1, E2):
        return shapely.hausdorff_distance(
            MultiLineString(E1), MultiLineString(E2), 0.99
        )

    def meshreport(self, mesh):
        """Print standard mesh report.  Valid in parallel."""
        nv, ne, hmin, hmax = self.meshsizes(mesh)
        PETSc.Sys.Print(
            f"current mesh: {nv} vertices, {ne} elements, h in [{hmin:.3f},{hmax:.3f}]"
        )
        return None

    # FIXME: checks for when free boundary is emptyset
    def freeboundarygraph(self, u, lb, type="coords"):
        """pulls the graph for the free boundary, return as dm, fd, or coords"""
        mesh = u.function_space().mesh()
        CellVertexMap = mesh.topology.cell_closure

        # Get active indicators
        nodalactive = self.nodalactive(u, lb)  # vertex
        elemactive = self.elemactive(u, lb)  # cell
        elemborder = self.elemborder(nodalactive)  # bordering cell

        # Pull Indices
        ActiveSetElementsIndices = [
            i for i, value in enumerate(elemactive.dat.data) if value != 0
        ]
        BorderElementsIndices = [
            i for i, value in enumerate(elemborder.dat.data) if value != 0
        ]

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
                for j in range(i + 1, len(vertices)):
                    v1 = vertices[i]
                    v2 = vertices[j]
                    # Add edge if both vertices are part of the free boundary
                    if v1 in FreeBoundaryVertices and v2 in FreeBoundaryVertices:
                        # Ensure consistent ordering
                        EdgeSet.add((min(v1, v2), max(v1, v2)))

        if type == "dm":
            return FreeBoundaryVertices, EdgeSet
        else:
            fdV = [
                mesh.topology._vertex_numbering.getOffset(vertex)
                for vertex in list(FreeBoundaryVertices)
            ]
            fdE = [
                [
                    mesh.topology._vertex_numbering.getOffset(edge[0]),
                    mesh.topology._vertex_numbering.getOffset(edge[1]),
                ]
                for edge in list(EdgeSet)
            ]
            if type == "fd":
                return fdV, fdE
            elif type == "coords":
                coords = mesh.coordinates.dat.data_ro_with_halos
                coordsV = [coords[vertex] for vertex in fdV]
                coordsE = [
                    [
                        [coords[edge[0]][0], coords[edge[0]][1]],
                        [coords[edge[1]][0], coords[edge[1]][1]],
                    ]
                    for edge in fdE
                ]
                return coordsV, coordsE
