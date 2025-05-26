import sys
import time
import numpy as np
from firedrake import *
from firedrake.petsc import OptionsManager, PETSc
from firedrake.output import VTKFile
from firedrake.utils import IntType
import firedrake.cython.dmcommon as dmcommon
from pyop2.mpi import MPI


class VIAMR(OptionsManager):
    """VIAMR object manages adaptive mesh refinement based on variational
    inequality concepts.  The central notion is that refinement near the
    free boundary will improve solution quality.  Those functions that
    do not work in parallel, namely udomark(), jaccard(), and hausdorff(), halt with
    ValueError on the size of the communicator for the mesh.  All other
    methods should work the same in serial and parallel."""

    def __init__(self, **kwargs):
        self.activetol = kwargs.pop("activetol", 1.0e-10)
        self.debug = kwargs.pop("debug", False)
        self.metricparameters = None

    def spaces(self, mesh, p=1):
        """Return CG{p} and DG{p-1} spaces."""
        if self.debug:
            assert isinstance(p, int)
            assert p >= 1
        return FunctionSpace(mesh, "CG", p), FunctionSpace(mesh, "DG", p - 1)

    def meshsizes(self, mesh):
        """Compute number of vertices, number of elements, and range of
        mesh diameters."""
        CG1, DG0 = self.spaces(mesh, p=1)
        nvertices = CG1.dim()
        nelements = DG0.dim()
        mymin, mymax = PETSc.INFINITY, PETSc.NINFINITY
        if len(mesh.cell_sizes.dat.data_ro) > 0:
            mymin = min(mesh.cell_sizes.dat.data_ro)
            mymax = max(mesh.cell_sizes.dat.data_ro)
        hmin = float(mesh.comm.allreduce(mymin, op=MPI.MIN))
        hmax = float(mesh.comm.allreduce(mymax, op=MPI.MAX))
        return nvertices, nelements, hmin, hmax

    def meshreport(self, mesh, indent=2):
        """Print standard mesh report."""
        nv, ne, hmin, hmax = self.meshsizes(mesh)
        indentstr = indent * " "
        PETSc.Sys.Print(
            f"{indentstr}current mesh: {nv} vertices, {ne} elements, h in [{hmin:.5f},{hmax:.5f}]"
        )
        return None

    def _checkadmissible(self, uh, bound, upper=False):
        if upper:
            bad = assemble(conditional(uh > bound, 1.0, 0.0) * dx)
        else:
            bad = assemble(conditional(uh < bound, 1.0, 0.0) * dx)
        return bad == 0.0

    def nodalactive(self, uh, lb):
        """Compute nodal active set indicator in same function space as uh.
        Applies to unilateral obstacle problems.  The active
        set is {x : u(x) == lb(x)}, within activetol.  Active nodes get value 1.0."""
        if self.debug:
            if len(uh.dat.data_ro) > 0 and len(lb.dat.data_ro) > 0:
                assert min(uh.dat.data_ro - lb.dat.data_ro) >= 0.0
            assert self._checkadmissible(uh, lb)
        z = Function(uh.function_space(), name="Nodal Active")
        z.interpolate(conditional(abs(uh - lb) < self.activetol, 1.0, 0.0))
        return z

    def elemactive(self, uh, lb):
        """Compute element active set indicator in DG0.  Applies to unilateral
        obstacle problems with.  Elements are marked active if the DG0 degree of
        freedom for that element is active, within activetol.  Active elements get
        value 1.0."""
        if self.debug:
            if len(uh.dat.data_ro) > 0 and len(lb.dat.data_ro) > 0:
                assert min(uh.dat.data_ro - lb.dat.data_ro) >= 0.0
            assert self._checkadmissible(uh, lb)
        _, DG0 = self.spaces(uh.function_space().mesh())
        z = Function(DG0, name="Element Active")
        z.interpolate(conditional(abs(uh - lb) < self.activetol, 1.0, 0.0))
        return z

    def eleminactive(self, uh, lb):
        """Compute element inactive set indicator in DG0.  Elements are marked
        inactive if the DG0 degree of freedom for that element is inactive, within
        activetol.  Inactive elements get value 1.0."""
        if self.debug:
            if len(uh.dat.data_ro) > 0 and len(lb.dat.data_ro) > 0:
                assert min(uh.dat.data_ro - lb.dat.data_ro) >= 0.0
            assert self._checkadmissible(uh, lb)
        _, DG0 = self.spaces(uh.function_space().mesh())
        z = Function(DG0, name="Element Inactive")
        z.interpolate(conditional(abs(uh - lb) < self.activetol, 0.0, 1.0))
        return z

    def elemborder(self, nodalactive):
        """From *nodal* active set indicator nodalactive, computes bordering
        element indicator.  Crucially uses the fact that the DG0 degree of
        freedom is strictly inside the element.  (Possibly only works if z is
        in CG1.)  Returns 1.0 for elements with
          0 < nodalactive(x_K) < 1
        at x_K which is the DG0 dof for element K."""
        if self.debug:
            if len(nodalactive.dat.data_ro) > 0:
                assert min(nodalactive.dat.data_ro) >= 0.0
                assert max(nodalactive.dat.data_ro) <= 1.0
        _, DG0 = self.spaces(nodalactive.function_space().mesh())
        z = Function(DG0, name="Element Border")
        z.interpolate(
            conditional(
                nodalactive > 0.0, conditional(nodalactive < 1.0, 1.0, 0.0), 0.0
            )
        )
        return z

    def unionmarks(self, mark1, mark2):
        """Computes the mark which is 1.0 where either mark1==1.0
        or mark2==1.0.  That is, computes the indicator set of the union."""
        return Function(mark1.function_space()).interpolate(
            (mark1 + mark2) - (mark1 * mark2)
        )

    def udomark(self, uh, lb, n=2):
        """Mark mesh using Unstructured Dilation Operator (UDO) algorithm.  The algorithm
        first computes an element-wise indicator for the free boundary.  Then the elements
        which neighbor free-boundary elements are added, and so on iteratively.
        The input n gives the number of levels to expand the initial element border.  The
        output is an element-wise marking for those elements near the free boundary which
        should be refined.
        Tuning advice:  Increase n to mark more elements near the free boundary, but
        on simple examples even n=1 may suffice."""

        # FIXME the size of the halo surely is connected to n?  so this won't run in
        #       parallel without communication or expanding the halo?  so this needs
        #       to use PetscSF?

        # get mesh and its DMPlex
        mesh = uh.function_space().mesh()
        d = mesh.cell_dimension()
        dm = mesh.topology_dm

        # Check that distribution parameters are correct in parallel
        if mesh.comm.size > 1:
            dp = mesh._distribution_parameters
            if dp["overlap_type"][0].name != "VERTEX" or dp["overlap_type"][1] < 1:
                raise ValueError(
                    """udomark() in parallel requires distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)} on mesh initialization."""
                )

        # This rest of this should really be written by turning the indicator function into a DMLabel
        # and then writing the dmplex traversal in petsc4py. This is a workaround.

        # Use nodal active set indicator to make a DG0 element border indicator, and then
        # convert to element indices.
        borderDG0 = self.elemborder(self.nodalactive(uh, lb))
        plexelementlist = mesh.cell_closure[:, -1]
        borderindices = [
            plexelementlist[k]
            for k, value in enumerate(borderDG0.dat.data_ro_with_halos)
            if value != 0
        ]

        # Find range of indices for element stratum
        kmin, kmax = dm.getHeightStratum(0)[:2]

        # main loop: expand element border out to n levels, using only DMPlex indices
        #   (index convention:  i for levels, j for nodes/vertices, k for elements)
        for i in range(n):
            # closure: Pull indices of all vertices which are incident
            # to some border element, then flatten and remove duplicates.
            incidentVertices = [
                dm.getTransitiveClosure(k)[0][-d-1:] for k in borderindices
            ]
            incidentVertices = np.unique(np.ravel(incidentVertices))

            # star: Pull indices of all elements which are incident to the
            # incidentVertices.  Note that getTransitiveClosure() with useCone=False
            # gives the star, and that the number of elements incident to a vertex
            # is not predictable.  Then flatten and remove duplicates.
            neighborindices = []
            for j in incidentVertices:
                star = dm.getTransitiveClosure(j, useCone=False)[0]
                mark = np.where((star >= kmin) & (star < kmax))
                neighborindices.extend(star[mark])
            borderindices = np.unique(np.ravel(neighborindices))

        # need map from DMPlex to firedrake indices
        # (Is there a better way to do this in dmcommon?)
        dm2fd = np.argsort(plexelementlist)

        # generate DG0 element border indicator by adding neighbors
        _, DG0 = self.spaces(mesh)
        border = Function(DG0).interpolate(Constant(0.0))
        for k in borderindices:
            border.dat.data_wo_with_halos[dm2fd[k]] = 1

        return border

    def vcdmark(
        self,
        uh,
        lb,
        bracket=[0.2, 0.8],
        coefficient=0.5,
        returnSmooth=False,
        directsolver=False,
        vcdsolveriters=4,
        printsolvertime=False,
    ):
        """Mark mesh using Variable Coefficient Diffusion (VCD) algorithm.
        The algorithm computes a nodal active set indicator and then
        diffuses it, using a variable mesh-sized based coefficient.  Diffusion
        is by solving a single backward Euler time step for the corresponding
        time-dependent diffusion equation.  Thresholding for the middle
        values of this field marks only those elements which are close to the
        free boundary.  The output is an element-wise marking for which elements,
        near the free boundary, should be refined.
        Tuning advice:  Generally the bracket [a,b] should be adjusted as follows:
          * lower a from default 0.2 to mark more elements in/near *inactive* set
          * raise b from default 0.8 to mark more elements in/near *active* set"""

        # Compute nodal active set indicator within some tolerance
        mesh = uh.function_space().mesh()
        CG1, DG0 = self.spaces(mesh)
        nu = self.nodalactive(uh, lb)

        # Diffuse according to square of cell diameter: D = C h^2.  The nodal
        # active indicator gives the initial field u0.  Then solve one backward
        # Euler time-step using a linear solver.
        w = TrialFunction(CG1)
        v = TestFunction(CG1)
        h = CellDiameter(mesh)
        a = w * v * dx + coefficient * h ** 2 * inner(grad(w), grad(v)) * dx
        L = nu * v * dx
        u = Function(CG1, name="Smoothed Nodal Active")

        if directsolver:
            sp = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        else:
            sp = {
                "ksp_type": "cg",
                "ksp_max_it": vcdsolveriters,
                "ksp_convergence_test": "skip",
                "pc_type": "icc",
            }
            if mesh.comm.size > 0:
                sp.update({"pc_type": "asm", "pc_asm_overlap": 1, "sub_pc_type": "icc"})
        if printsolvertime:
            start = time.perf_counter()
        solve(a == L, u, solver_parameters=sp, options_prefix="viamr_vcd")
        if printsolvertime:
            end = time.perf_counter()
            PETSc.Sys.Print(
                f"VIAMR INFO  vcdmark() solver time = {end - start:.6f} seconds"
            )

        if returnSmooth:
            return u

        # apply thresholding and interpolate into DG0
        mark = Function(DG0, name="VCD Marking")
        mark.interpolate(
            conditional(u > bracket[0], conditional(u < bracket[1], 1, 0), 0)
        )
        return mark

    def _fixedrate(self, eta, theta, method):
        """Marks elements according to the values of eta and a threshold which
        depends on theta.  The number of elements marked is an increasing function
        of theta.  The default 'max' strategy marks all elements with eta greater
        than
          ethresh = theta * max eta
        The 'total' strategy sorts the elements owned by the process by decreasing
        eta value.  Then the threshold
          ethresh = eta(index)
        equals the eta value where theta times the total sum of eta is equal to the
        sum of the eta values above ethresh.  (I.e. theta gives the fraction of the
        total eta sum.)  The 'total' strategy is the refine-only version of the
        "fixed-rate" strategy, with X=theta and Y=0, described in section 4.2 of
          W. Bangerth & R. Rannacher (2003).  Adaptive Finite Element Methods for
          Differential Equations, Springer Basel.
        WARNING: The 'total' strategy produces different results depending on
        the number of processes."""

        with eta.dat.vec_ro as eta_:
            if method == "max":
                ethresh = theta * eta_.max()[1]
            elif method == "total":
                values = eta_.array_r
                sorted_values = np.sort(values)[::-1]  # sort in descending order
                cumsum = np.cumsum(sorted_values)
                target = np.sum(values) * theta  # proportion of total error
                idx = np.argmax(cumsum >= target)
                ethresh = sorted_values[idx]
            else:
                raise ValueError("unknown method for VIAMR._fixedrate()")
            total_error_est = sqrt(eta_.dot(eta_))

        DG0 = eta.function_space()
        mark = Function(DG0).interpolate(conditional(gt(eta, ethresh), 1, 0))
        return mark, ethresh, total_error_est

    def gradrecinactivemark(self, uh, lb, theta=0.5, method="max"):
        """Return marking within the computed inactive set by using an
        a posteriori gradient-recovery error indicator.  See Chapter 4 of
          M. Ainsworth & J. T. Oden (2000).  A Posteriori Error Estimation in
          Finite Element Analysis, John Wiley & Sons, Inc., New York."""
        mesh = uh.function_space().mesh()
        v = CellVolume(mesh)
        # recover a CG1 gradient of uh by projection
        CG1vec = VectorFunctionSpace(mesh, "CG", 1)
        gradrecu = Function(CG1vec).project(grad(uh))
        # cell-wise error estimator
        _, DG0 = self.spaces(mesh)
        eta_sq = Function(DG0)
        w = TestFunction(DG0)
        G = (
            inner(eta_sq / v, w) * dx
            - inner(inner(gradrecu - grad(uh), gradrecu - grad(uh)), w) * dx
        )
        # each cell needs an independent 1x1 solve, so Jacobi is an exact preconditioner
        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)
        eta = Function(DG0).interpolate(sqrt(eta_sq))  # eta from eta^2
        # restrict grad recovery eta to inactive set
        imark = self.eleminactive(uh, lb)
        ieta = Function(DG0, name="eta on inactive set").interpolate(eta * imark)
        # compute mark in inactive set
        mark, _, total_error_est = self._fixedrate(ieta, theta, method)
        return (mark, eta, total_error_est)

    def brinactivemark(self, uh, lb, res_ufl, theta=0.5, method="max"):
        """Return marking within the computed inactive set by using the
        a posteriori Babuška-Rheinboldt residual error indicator.  The BR
        indicator eta is computed as a function in DG0.  Then we call
        VIAMR._fixedrate() to mark using eta and a threshold theta.
        Returns the marking mark, estimator eta, and a scalar estimate for
        the total error in energy norm.  (This last value is only valid for
        the Poisson equation.)  This function is on slide 109 of
          https://github.com/pefarrell/icerm2024/blob/main/slides.pdf
        See also
          https://github.com/pefarrell/icerm2024/blob/main/02_netgen/01_l_shaped_adaptivity.py
        and section 2.2 of
          M. Ainsworth & J. T. Oden (2000).  A Posteriori Error Estimation in
          Finite Element Analysis, John Wiley & Sons, Inc., New York."""
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
            inner(eta_sq / v, w) * dx
            - inner(h ** 2 * res_ufl ** 2, w) * dx
            - inner(h("+") / 2 * jump(grad(uh), n) ** 2, w("+")) * dS
            - inner(h("-") / 2 * jump(grad(uh), n) ** 2, w("-")) * dS
        )
        # each cell needs an independent 1x1 solve, so Jacobi is an exact preconditioner
        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)
        eta = Function(DG0).interpolate(sqrt(eta_sq))  # eta from eta^2
        # restrict BR eta to inactive set
        imark = self.eleminactive(uh, lb)
        ieta = Function(DG0, name="eta on inactive set").interpolate(eta * imark)
        mark, _, total_error_est = self._fixedrate(ieta, theta, method)
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

    def metricfromhessian(self, uh):
        """Construct a hessian based metric from a solution"""

        assert (
            self.metricparameters is not None
        ), "call setmetricparameters() before calling metricfromhessian()"

        from animate import RiemannianMetric

        mesh = uh.function_space().mesh()
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)
        metric.set_parameters(self.metricparameters)
        metric.compute_hessian(uh)
        metric.normalise()
        return metric

    def metricrefine(self, mesh, uh, lb, weights=[0.50, 0.50], hessian=True):
        """Implementation of anisotropic metric based refinement which is
        free-boundary aware. Constructs both the hessian based metric and an
        isotropic metric computed from the magnitude of the gradient of the
        smoothed VCD indicator.  These metrics are averaged using the weights."""

        assert (
            self.metricparameters is not None
        ), "call setmetricparameters() before calling metricrefine()"

        from animate import adapt, RiemannianMetric

        # Construct isotropic metric from abs(grad(smoothed_vcd_indicator))
        V, _ = self.spaces(mesh)
        dim = mesh.topological_dimension()

        # Get magnitude of gradients
        s = self.vcdmark(uh, lb, returnSmooth=True)
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
            solutionMetric = self.metricfromhessian(uh)
            solutionMetric.normalise()
            VImetric.average(freeboundaryMetric, weights=weights)
        else:
            VImetric = freeboundaryMetric.copy(deepcopy=True)

        return adapt(mesh, VImetric)

    def refinemarkedelements(self, mesh, indicator, isUniform=False):
        """petsc4py implementation of Netgen's .refine_marked_elements().
        Usually is skeleton-based refinement (SBR; Plaza & Carey, 2000).
        This version works in parallel, but only in 2D when using SBR;
        see TODO in
          https://petsc.org/release/src/dm/impls/plex/transform/impls/refine/sbr/plexrefsbr.c.html.
        See also
          https://petsc.org/release/overview/plex_transform_table/
        and associated links."""
        # Create Section for DG0 indicator
        tdim = mesh.topological_dimension()
        entity_dofs = np.zeros(tdim + 1, dtype=IntType)
        entity_dofs[:] = 0
        entity_dofs[-1] = 1
        indicatorSect, _ = dmcommon.create_section(mesh, entity_dofs)

        # Pull Plex from mesh
        dm = mesh.topology_dm

        # Create an adaptation label to mark cells for refinement
        dm.createLabel("refinesbr")
        adaptLabel = dm.getLabel("refinesbr")
        adaptLabel.setDefaultValue(0)

        # dmcommon provides a python binding for this operation of setting
        # the label given an indicator function data array
        dmcommon.mark_points_with_function_array(
            dm, indicatorSect, 0, indicator.dat.data_with_halos, adaptLabel, 1
        )

        # augment or override options database to indicate type of refinement
        # For now the only way to set the active label with petsc4py is
        # with PETSc.Options() because DMPlexTransformSetActive() has no binding.
        opts = PETSc.Options()
        if isUniform:
            opts["dm_plex_transform_type"] = "refine_regular"
        else:
            opts["dm_plex_transform_active"] = "refinesbr"
            opts[
                "dm_plex_transform_type"
            ] = "refine_sbr"  # <- skeleton based refinement is what netgen does.

        # Create a DMPlexTransform object to apply the refinement
        dmTransform = PETSc.DMPlexTransform().create(comm=mesh.comm)
        dmTransform.setDM(dm)
        dmTransform.setFromOptions()
        dmTransform.setUp()
        dmAdapt = dmTransform.apply(dm)

        # Labels are no longer needed, not sure if we need to call destroy on them.
        dmAdapt.removeLabel("refinesbr")
        dm.removeLabel("refinesbr")
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
        refinedmesh = Mesh(dmAdapt, distribution_parameters=distParams, comm=mesh.comm)
        opts["dm_plex_transform_type"] = "refine_regular"

        return refinedmesh

    def jaccard(self, active1, active2, submesh=False):
        """Compute the Jaccard metric from two element-wise active set
        indicators.  By definition, the Jaccard metric of two sets is
            J(S,T) = |S cap T| / |S cup T|,
        where |.| is area (measure) of the set.  Thus J(S,T) the ratio of
        the area (measure) of the intersection divided by that of the union.
        The inputs are the indicator functions of the sets as DG0 functions.
        In serial they can be on different meshes.  (In that case project()
        method is used to put them on active1's mesh.)  If submesh==True
        then active2 is assumed to live on a submesh of active1, so
        interpolate onto the active1 mesh will work correctly.
        *Thus with submesh==True it works in parallel.*"""
        # FIXME how to check that active1, active2 are in DG0 spaces?
        # FIXME how to check that, when submesh==True, active2 is actually
        #       on a submesh of active1?
        if submesh == False:
            if (
                active1.function_space().mesh()._comm.size > 1
                or active2.function_space().mesh()._comm.size > 1
            ):
                raise ValueError("jaccard(.., submesh=False) is not valid in parallel")
        if self.debug:
            for a in [active1, active2]:
                if len(a.dat.data_ro) > 0:
                    assert min(a.dat.data_ro) >= 0.0
                    assert max(a.dat.data_ro) <= 1.0
        mesh1 = active1.function_space().mesh()
        if submesh:
            new2 = Function(active1.function_space()).interpolate(active2)
        else:
            new2 = Function(active1.function_space()).project(active2)
        AreaIntersection = assemble(new2 * active1 * dx(mesh1))
        AreaUnion = assemble((new2 + active1 - (new2 * active1)) * dx(mesh1))
        assert AreaUnion > 0.0, "jaccard() computed measure of the union as zero"
        return AreaIntersection / AreaUnion

    def hausdorff(self, E1, E2):
        try:
            import shapely
        except ImportError:
            print("ImportError.  Unable to import shapely.  Exiting.")
            sys.exit(0)
        return shapely.hausdorff_distance(
            shapely.MultiLineString(E1), shapely.MultiLineString(E2), 0.99
        )

    # FIXME: checks for when free boundary is emptyset
    def freeboundarygraph(self, uh, lb, type="coords"):
        """pulls the graph for the free boundary, return as dm, fd, or coords"""
        mesh = uh.function_space().mesh()
        CellVertexMap = mesh.topology.cell_closure

        # Get active indicators
        elemactive = self.elemactive(uh, lb)  # cell
        elemborder = self.elemborder(self.nodalactive(uh, lb))  # bordering cell

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
