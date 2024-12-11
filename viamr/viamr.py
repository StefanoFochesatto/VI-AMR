import numpy as np
from collections import deque
from firedrake import *
from firedrake.petsc import OptionsManager, PETSc
from firedrake.output import VTKFile
from pyop2.mpi import MPI


class VIAMR(OptionsManager):

    def __init__(self, **kwargs):
        # solver_parameters = flatten_parameters(kwargs.pop("solver_parameters",{}))
        # self.options_prefix = kwargs.pop("options_prefix","")
        # super().__init__(solver_parameters, self.options_prefix)  # set PETSc parameters
        self.activetol = 1.0e-10

    def spaces(self, mesh, p=1):
        '''Return CG{p} and DG{p-1} spaces.'''
        return FunctionSpace(mesh, "CG", p), FunctionSpace(mesh, "DG", p-1)

    def meshsizes(self, mesh):
        '''Compute number of vertices, number of elements, and range of
        mesh diameters.  Valid in parallel.'''
        CG1, DG0 = self.spaces(mesh, p=1)
        nvertices = CG1.dim()
        nelements = DG0.dim()
        hmin = float(mesh.comm.allreduce(min(mesh.cell_sizes.dat.data_ro), op=MPI.MIN))
        hmax = float(mesh.comm.allreduce(max(mesh.cell_sizes.dat.data_ro), op=MPI.MAX))
        return nvertices, nelements, hmin, hmax

    def meshreport(self, mesh, indent=2):
        '''Print standard mesh report.  Valid in parallel.'''
        nv, ne, hmin, hmax = self.meshsizes(mesh)
        PETSc.Sys.Print(f'current mesh: {nv} vertices, {ne} elements, h in [{hmin:.3f},{hmax:.3f}]')
        return None

    def nodalactive(self, u, lb):
        '''Compute nodal active set indicator in same function space as u.'''
        z = Function(u.function_space(), name="Nodal Active Set Indicator")
        z.interpolate(conditional(abs(u - lb) < self.activetol, 0, 1))
        return z

    def bfs_neighbors(self, cmesh, border, levels, active):
        '''element-wise Fast Multi Neighbor Lookup BFS can Avoid Active Set'''

        # dictionary to map each vertex to the cells that contain it
        vertex_to_cells = {}
        cell_vertex_map = cmesh.topology.cell_closure # cell to vertex connectivity
        # Loop over all cells to populate the dictionary
        for i in range(cmesh.num_cells()):
            # first three entries correspond to the vertices
            for vertex in cell_vertex_map[i][:3]:
                if vertex not in vertex_to_cells:
                    vertex_to_cells[vertex] = []
                vertex_to_cells[vertex].append(i)

        # Loop over all cells to mark neighbors
        # Create a new DG0 function to store the result
        result = Function(border.function_space(), name='nNeighbors')
        for i in range(cmesh.num_cells()):
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

    def udomark(self, cmesh, u, lb, n=2):
        "Mark current mesh using Unstructured Dilation Operator (UDO) algorithm."

        # generate element-wise and nodal-wise indicators for active set
        CG1, DG0 = self.spaces(cmesh)
        nodaldiffCG = Function(CG1).interpolate(abs(u - lb))
        elemactive = Function(DG0, name="Element Active Set Indicator")
        elemactive.interpolate(conditional(nodaldiffCG < self.activetol, 1, 0))
        nodalactive = self.nodalactive(u, lb)

        # Define Border Elements Set
        elemborder = Function(DG0, name="Border Elements")
        elemborder.interpolate(conditional(nodalactive > 0,
                                           conditional(nodalactive < 1, 1, 0),
                                           0))

        # bfs_neighbors() constructs N^n(B) indicator.  Last argument
        # is to refine only in active or only in inactive set (currently commented out).
        return self.bfs_neighbors(cmesh, elemborder, n, elemactive)


    def vcesmark(self, cmesh, u, lb, bracket=[0.2, 0.8]):
        "Mark current mesh using Variable Coefficient Elliptic Smoothing (VCES) algorithm"

        # Compute nodal active set indicator within some tolerance
        CG1, DG0 = self.spaces(cmesh)
        nodalactive = self.nodalactive(u, lb)

        # Vary timestep by average cell area of each patch.
        # Not exactly an average because msh.cell_sizes is an L2 projection of
        # the obvious DG0 function into CG1.
        timestep = Function(CG1)
        timestep.dat.data[:] = 0.5 * cmesh.cell_sizes.dat.data[:] ** 2

        # Solve one step implicitly using a linear solver
        # Nodal indicator is initial condition to time dependent Heat eq
        w = TrialFunction(CG1)
        v = TestFunction(CG1)
        a = w * v * dx + timestep * inner(grad(w), grad(v)) * dx
        L = nodalactive * v * dx
        u = Function(CG1, name="Smoothed Nodal Active Indicator")
        solve(a == L, u)

        # Compute average over elements by interpolation into DG0
        DG0 = FunctionSpace(cmesh, "DG", 0)
        uDG0 = Function(DG0, name="Smoothed Nodal Active Indicator as DG0")
        uDG0.interpolate(u)

        # Applying thresholding parameters
        mark = Function(DG0, name="VCES Marking")
        mark.interpolate(
            conditional(uDG0 > bracket[0], conditional(uDG0 < bracket[1], 1, 0), 0)
        )
        return mark
