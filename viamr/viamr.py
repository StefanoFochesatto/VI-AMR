import numpy as np
from collections import deque

from firedrake import *
from firedrake.petsc import OptionsManager, PETSc
from firedrake.output import VTKFile
from firedrake.utils import IntType
import firedrake.cython.dmcommon as dmcommon

from pyop2.mpi import MPI

try:
    import shapely
except ImportError:
    print("ImportError.  Unable to import shapely.  Exiting.")
    import sys
    sys.exit(0)


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

    def meshreport(self, mesh, indent=2):
        """Print standard mesh report.  Valid in parallel."""
        nv, ne, hmin, hmax = self.meshsizes(mesh)
        indentstr = indent * ' '
        PETSc.Sys.Print(
            f"{indentstr}current mesh: {nv} vertices, {ne} elements, h in [{hmin:.5f},{hmax:.5f}]"
        )
        return None

    def nodalactive(self, u, lb):
        """Compute nodal active set indicator in same function space as u.
        Applies to unilateral obstacle problems with u >= lb.  The active
        set is {x : u(x) == lb(x)}, within activetol."""
        if self.debug:
            assert min(u.dat.data_ro - lb.dat.data_ro) >= 0.0
        z = Function(u.function_space(), name="Nodal Active")
        z.interpolate(conditional(abs(u - lb) < self.activetol, 0, 1))
        return z

    def elemactive(self, u, lb):
        """Compute element active set indicator in DG0."""
        if self.debug:
            assert min(u.dat.data_ro - lb.dat.data_ro) >= 0.0
        _, DG0 = self.spaces(u.function_space().mesh())
        z = Function(DG0, name="Element Active")
        z.interpolate(conditional(abs(u - lb) < self.activetol, 1, 0))
        return z
    
    def eleminactive(self, u, lb):
        """Compute element inactive set indicator in DG0."""
        if self.debug:
            assert min(u.dat.data_ro - lb.dat.data_ro) >= 0.0
        _, DG0 = self.spaces(u.function_space().mesh())
        z = Function(DG0, name="Element Inactive")
        z.interpolate(conditional(abs(u - lb) < self.activetol, 0, 1))
        return z

    def elemborder(self, nodalactive):
        """From nodal activeset indicator compute bordering element indicator."""
        if self.debug:
            assert min(nodalactive.dat.data_ro) >= 0.0
            assert max(nodalactive.dat.data_ro) <= 1.0
        _, DG0 = self.spaces(nodalactive.function_space().mesh())
        z = Function(DG0, name="Element Border")
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
    
    
    def udomarkParallel(self, mesh, u, lb, n=2):
        '''Mark mesh using Unstructured Dilation Operator (UDO) algorithm. Update to latest ngsPETSc otherwise refinement must be done with PETSc refinemarkedelements'''
        
        # Generate element-wise and nodal-wise indicators for active set
        _, DG0 = self.spaces(mesh)
        nodalactive = self.nodalactive(u, lb)

        # Generate border element indicator
        elemborder = self.elemborder(nodalactive)
        
        mesh.name = 'dmmesh'
        elemborder.rename('elemborder')

        # Checkpointing to enforce distribution parameters which make UDO possible in parallel
        # This workaround is necessary because:
        # 1. firedrake does not have a way of changing distribution parameters after mesh initialization (feature request)
        # 2. netgen meshes cannot be checkpointed in parallel (issue)
        # 
        # In order for this to work we need to use PETSc refinemarkelements instead
        # also instead of checkpointing we could write a warning telling the user to set the correct distribution parameters
        # 
        
        
        DistParams = mesh._distribution_parameters
        
        if DistParams['overlap_type'][0].name != 'VERTEX' or DistParams['overlap_type'][1] < 1:
            #We will error out instead
            raise ValueError("""Error: For UDO to work ensure that distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)} on mesh initialization.""")
            
            # This workaround works for firedrake meshes, not netgen. It also forces me to return the mesh which is a bad user pattern. 
            # MPI.COMM_WORLD.Barrier()
            # PETSc.Sys.Print("entered bad params")
            # PETSc.Sys.Print("writing")
            # with CheckpointFile("udo.h5", 'w') as afile:
            #     afile.save_mesh(mesh)
            #     afile.save_function(elemborder)
            # PETSc.Sys.Print("writing finished")
            # PETSc.Sys.Print("reading")
            # with CheckpointFile("udo.h5", 'r') as afile:
            #     mesh = afile.load_mesh("dmmesh", distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}) # <- enforcing distribution parameters
            #     elemborder = afile.load_function(mesh, "elemborder")
            # PETSc.Sys.Print("reading finished")


            # # reconstruct DG0 space so result indicator has correct partition    
            # _, DG0 = self.spaces(mesh)
    


        # Pull dm 
        dm = mesh.topology_dm
        
        # This rest of this should really be written by turning the indicator function into a DMLabel
        # and then writing the dmplex traversal in petsc4py. This is a workaround.
        
        
        # Generate map from dm to fd indices (I think there is a better way to do this in dmcommon)
        plexelementlist = mesh.cell_closure[:, -1]
        dm_to_fd = {number: index for index,
                    number in enumerate(plexelementlist)}

        for i in range(n):
            # Pull border elements cell with dmplex cell indices
            BorderSetElementsIndices = [mesh.cell_closure[:, -1][i] for i, value in enumerate(
                elemborder.dat.data_ro_with_halos) if value != 0]



            # Pull indices of vertices which are incident to said border elements
            incidentVertices = [dm.getTransitiveClosure(
                i)[0][4:7] for i in BorderSetElementsIndices]

     
            # Flatten the list of lists and remove duplicates
            flattened_array = np.ravel(incidentVertices)
            incidentVertices = np.unique(flattened_array)

            # Needs to be based of topological dimension
            # Pull the depth stratum for the vertices           
            tdim = mesh.topological_dimension() 
            lb = dm.getDepthStratum(tdim)[0]
            ub = dm.getDepthStratum(tdim)[1]
            # Pull all elements which are neighbor to the incidentVertices. This produces the set N(B)
            NeighborSet = []
            for i in incidentVertices:
                idx = np.where((dm.getTransitiveClosure(i, useCone=False)[0] >= lb) & (dm.getTransitiveClosure(i, useCone=False)[0] < ub))
                NeighborSet.extend(dm.getTransitiveClosure(i, useCone=False)[0][idx])
            # Flatten the list of lists and remove duplicates
            NeighborSet = np.ravel(NeighborSet)
            NeighborSet = np.unique(NeighborSet)


            # Create new elemborder function
            elemborder = Function(DG0).interpolate(Constant(0.0))

            for j in NeighborSet:
                elemborder.dat.data_wo_with_halos[dm_to_fd[j]] = 1
        
        return elemborder


    def vcdmark(self, mesh, u, lb, bracket=[0.2, 0.8], returnSmooth=False):
        """Mark mesh using Variable Coefficient Diffusion (VCD) algorithm.
        Conceptually, the algorithm computes a strict nodal active set
        indicator and then it diffuses using variable diffusivity using a
        single backward Euler time step for the heat equation.  (Equivalently
        we take a single time-step of variable duration.) Valid in parallel."""

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
        u = Function(CG1, name="Smoothed Nodal Active")
        #FIXME consider some solver; probably not this one: sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(a == L, u)

        if returnSmooth:
            return u
        else:
            # Compute average over elements by interpolation into DG0
            DG0 = FunctionSpace(mesh, "DG", 0)
            uDG0 = Function(DG0, name="Smoothed Nodal Active as DG0")
            uDG0.interpolate(u)
            # Applying thresholding parameters
            mark = Function(DG0, name="VCD Marking")
            mark.interpolate(
                conditional(uDG0 > bracket[0], conditional(uDG0 < bracket[1], 1, 0), 0)
            )
            return mark

    def br_mark_poisson(self, uh, lb, f=Constant(0.0), theta=0.5):
        '''Return marking within the computed inactive set by using the a posteriori
        Babuška-Rheinboldt residual error indicator for the Poisson equation
          - div(grad u) = f
        First the BR indicator eta is computed as a function in DG0.  Then
        we mark where eta is larger than theta fraction of eta values.
        Returns the marking mark, estimator eta, and a scalar estimate for
        the total error in energy norm.  This function is on slide 109 of
          https://github.com/pefarrell/icerm2024/blob/main/slides.pdf
        The original source is perhaps
          Babuška, I., & Rheinboldt, W. C. (1978). Error estimates for adaptive
          finite element computations. SIAM Journal on Numerical Analysis,
          15(4), 736-754.  https://www.jstor.org/stable/pdf/2156851.pdf
        FIXME this should be more flexible, and at least correspond to
          - A div(grad u) = f
        for A>0'''
        # mesh quantities
        mesh = uh.function_space().mesh()
        h = CellDiameter(mesh)
        v = CellVolume(mesh)
        n = FacetNormal(mesh)
        # cell-wise error estimator
        _, DG0 = self.spaces(mesh)
        eta_sq = Function(DG0)
        w = TestFunction(DG0)
        G = (
            inner(eta_sq / v, w)*dx
            - inner(h**2 * (f + div(grad(uh)))**2, w) * dx
            - inner(h('+')/2 * jump(grad(uh), n)**2, w('+')) * dS
            - inner(h('-')/2 * jump(grad(uh), n)**2, w('-')) * dS
        )
        # Each cell is an independent 1x1 solve, so Jacobi is an exact preconditioner
        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)
        eta = Function(DG0).interpolate(sqrt(eta_sq))  # eta from eta^2
        # generate inactive BR marking
        imark = self.eleminactive(uh, lb)
        ieta = Function(DG0, name='eta on inactive set').interpolate(eta * imark)
        with ieta.dat.vec_ro as ieta_:
            emax = ieta_.max()[1]
            total_error_est = sqrt(ieta_.dot(ieta_))
        mark = Function(DG0).interpolate(conditional(gt(ieta, theta * emax), 1, 0))
        return (mark, eta, total_error_est)

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


    def metricrefine(self, mesh, u, lb, weights=[0.50, 0.50], hessian = True):
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
        if hessian:
            VImetric = freeboundaryMetric.copy(deepcopy=True)
            solutionMetric = self.metricfromhessian(mesh, u)
            solutionMetric.normalise()
            VImetric.average(freeboundaryMetric, weights=weights)
        else:
            VImetric = freeboundaryMetric.copy(deepcopy=True)

        return VImetric
    

    # Computes Babuška Rheinboldt(1978) error estimator, returns marking function on only inactive set.
    # Error estimator computation is from
    #   https://github.com/pefarrell/icerm2024/blob/main/slides.pdf  (slide 109)
    #   https://github.com/pefarrell/icerm2024/blob/main/02_netgen/01_l_shaped_adaptivity.py
    def BRinactivemark(self, mesh, u, lb, resUFL, theta, markFB = None):
        _ , W = self.spaces(mesh)
        eta_sq = Function(W)
        w = TestFunction(W)
        
        h = CellDiameter(mesh)  # symbols for mesh quantities
        n = FacetNormal(mesh)
        v = CellVolume(mesh)

        # Babuska-Rheinboldt error estimator
        G = (  # compute cellwise error estimator
            inner(eta_sq / v, w)*dx
            - inner(h**2 * (resUFL)**2, w) * dx
            - inner(h('+')/2 * jump(grad(u), n)**2, w('+')) * dS
            - inner(h('-')/2 * jump(grad(u), n)**2, w('-')) * dS
        )

        # Each cell is an independent 1x1 solve, so Jacobi is exact
        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)
        eta = Function(W).interpolate(sqrt(eta_sq))  # compute eta from eta^2

        # Thinking about removing the Free Boundary mark from here. 
        # Mask inactive set 
        eleminactive = self.eleminactive(u, lb)
        if markFB == None:
            etaInactive = Function(W).interpolate(eta * eleminactive)
        else:
            etaInactive = Function(W).interpolate(eta * eleminactive* abs(markFB - 1))
            
        
        # Refine all cells greater than theta*eta_max
        with etaInactive.dat.vec_ro as eta_:
            eta_max = eta_.max()[1]        
        refine = conditional(gt(etaInactive, theta*eta_max), 1, 0)
        mark = Function(W).interpolate(refine)
        
        return mark
            
            
    def union(self, mark1, mark2):
        markUnion = Function(mark1.function_space()).interpolate((mark1 + mark2) - (mark1*mark2))
        return markUnion
        
    
    def refinemarkedelements(self, mesh, indicator, isUniform = False):
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
        dm.createLabel('refinesbr')
        adaptLabel = dm.getLabel('refinesbr')
        adaptLabel.setDefaultValue(0)

        # dmcommon provides a python binding for this operation of setting the label given an indicator function data array
        dmcommon.mark_points_with_function_array(
            dm, indicatorSect, 0, indicator.dat.data_with_halos, adaptLabel, 1)

        # Create a DMPlexTransform object to apply the refinement
        opts = PETSc.Options()
        if isUniform:
            opts['dm_plex_transform_type'] = 'refine_regular'
        else:
            opts['dm_plex_transform_active'] = 'refinesbr'
            opts['dm_plex_transform_type'] = 'refine_sbr' # <- skeleton based refinement is what netgen does.
        dmTransform = PETSc.DMPlexTransform().create(comm = mesh.comm)
        dmTransform.setDM(dm)
        # For now the only way to set the active label with petsc4py is with PETSc.Options() (DMPlexTransformSetActive() has no binding)
        dmTransform.setFromOptions()
        dmTransform.setUp()
        dmAdapt = dmTransform.apply(dm)
        
        # Labels are no longer needed, not sure if we need to call destroy on them. 
        dmAdapt.removeLabel('refinesbr')
        dm.removeLabel('refinesbr')
        dmTransform.destroy()
        
        # Remove labels to stop further distribution in mesh()
        # dm.distributeSetDefault(False) <- Matt's suggestion
        dmAdapt.removeLabel("pyop2_core")
        dmAdapt.removeLabel("pyop2_owned")
        dmAdapt.removeLabel("pyop2_ghost")
        # ^ Koki's suggestion
    
        # Pull distribution parameters from original dm
        distParams = mesh._distribution_parameters
        
        # Create a new mesh from the adapted dm
        refinedmesh = Mesh(dmAdapt, distribution_parameters = distParams, comm = mesh.comm)
        opts['dm_plex_transform_type'] = 'refine_regular'
        
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
            shapely.MultiLineString(E1), shapely.MultiLineString(E2), 0.99
        )

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
