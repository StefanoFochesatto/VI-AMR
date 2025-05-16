import numpy as np
import pytest
from firedrake import *
from viamr import VIAMR
import subprocess
import os
import pathlib

from test_basic import _get_netgen_mesh, _get_ball_obstacle

from mpi4py import MPI
from pytest_mpi.parallel_assert import parallel_assert
from pytest_mpi import parallel_assert


class VIAMRRegression(VIAMR):
    def __init__(self):
        super().__init__()

    def _bfsneighbors(self, mesh, border, levels):
        """Element-wise multi-neighbor lookup using breadth-first search."""
        from collections import deque

        # build dictionary which maps each vertex in the mesh
        # to the cells that are incident to it
        vertex_to_cells = {}
        closure = mesh.topology.cell_closure  # cell to vertex connectivity
        # loop over all cells to populate the dictionary
        for i in range(mesh.num_cells()):
            # first three entries correspond to the vertices
            for vertex in closure[i][:3]:
                if vertex not in vertex_to_cells:
                    vertex_to_cells[vertex] = []
                vertex_to_cells[vertex].append(i)

        # loop over all cells to mark neighbors, and store the result in DG0
        result = Function(border.function_space(), name="nNeighbors")
        for i in range(mesh.num_cells()):
            if border.dat.data[i] == 1.0:
                # use BFS to find all cells within the specified number of levels
                queue = deque([(i, 0)])
                visited = set()
                while queue:
                    cell, level = queue.popleft()
                    if cell not in visited and level <= levels:
                        visited.add(cell)
                        result.dat.data[cell] = 1
                        for vertex in closure[cell][:3]:
                            for neighbor in vertex_to_cells[vertex]:
                                queue.append((neighbor, level + 1))
        return result

    def udomarkOLD(self, uh, lb, n=2):
        """Mark mesh using Unstructured Dilation Operator (UDO) algorithm."""
        mesh = uh.function_space().mesh()
        if mesh.comm.size > 1:
            raise ValueError("udomark() is not valid in parallel")
        # generate element-wise indicator for border set
        elemborder = self.elemborder(self.nodalactive(uh, lb))
        # _bfs_neighbors() constructs N^n(B) indicator
        return self._bfsneighbors(mesh, elemborder, n)


def PARtest_refine_udo_parallelUDO():
    mesh1 = _get_netgen_mesh(TriHeight=0.1)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh1)
    (x, y) = SpatialCoordinate(mesh1)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    # VTKFile(f"result_refine_0.pvd").write(u)
    mark1 = amr.udomark(u, psi)
    rmesh1 = mesh1.refine_marked_elements(mark1)  # netgen's refine method
    mesh2 = _get_netgen_mesh(TriHeight=0.1)
    CG1, _ = amr.spaces(mesh2)
    (x, y) = SpatialCoordinate(mesh1)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    # VTKFile("result_refine_0.pvd").write(u)
    mark2 = amr.udomark(u, psi)
    rmesh2 = mesh2.refine_marked_elements(mark2)  # netgen's refine method
    assert abs(amr.jaccard(mark1, mark2) - 1.0) < 1.0e-10
    r1CG1, _ = amr.spaces(rmesh1)
    r2CG1, _ = amr.spaces(rmesh2)
    assert r1CG1.dim() == r2CG1.dim()


@pytest.mark.parallel(nprocs=3)
def PARtest_parallel_udo():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    amr = VIAMR()
    mesh = RectangleMesh(
        20,
        20,
        1,
        1,
        distribution_parameters={
            "partition": True,
            "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
        },
    )
    CG1, _ = amr.spaces(mesh)
    u = Function(CG1).interpolate(1.0)
    x, y = SpatialCoordinate(mesh)

    # Somewhat interesting obstacle configuration.
    lb = Function(CG1).interpolate(
        conditional(
            And(And(x > 0.15, x < 0.35), And(y > 0.15, y < 0.35)),
            1.0,
            conditional(
                And(And(x > 0.65, x < 0.85), And(y > 0.15, y < 0.35)),
                1.0,
                conditional(
                    And(And(x > 0.15, x < 0.35), And(y > 0.65, y < 0.85)),
                    1.0,
                    conditional(
                        And(And(x > 0.65, x < 0.85), And(y > 0.65, y < 0.85)), 1.0, 0.0
                    ),
                ),
            ),
        )
    )

    # Mark in parallel
    mark = amr.udomark(u, lb, n=2)

    # Compute number of local active elements
    localActive = np.sum(mark.dat.data_ro[:])
    globalActive = np.zeros(1, dtype=np.float64)
    comm.Allreduce(localActive, globalActive, op=MPI.SUM)

    if rank == 0:
        # Check agreement with serial implementation
        assert globalActive[0] == 506


def PARtest_udo_regression():
    # This test utilizes the the old implementation of UDO which builds the neighborhood of the free boundary using breadth first search,
    # as a regression test for the dmplex based implementation.

    amr = VIAMRRegression()
    mesh = RectangleMesh(
        20,
        20,
        1,
        1,
        distribution_parameters={
            "partition": True,
            "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
        },
    )
    CG1, _ = amr.spaces(mesh)
    u = Function(CG1).interpolate(1.0)
    x, y = SpatialCoordinate(mesh)

    # Somewhat interesting obstacle configuration.
    lb = Function(CG1).interpolate(
        conditional(
            And(And(x > 0.15, x < 0.35), And(y > 0.15, y < 0.35)),
            1.0,
            conditional(
                And(And(x > 0.65, x < 0.85), And(y > 0.15, y < 0.35)),
                1.0,
                conditional(
                    And(And(x > 0.15, x < 0.35), And(y > 0.65, y < 0.85)),
                    1.0,
                    conditional(
                        And(And(x > 0.65, x < 0.85), And(y > 0.65, y < 0.85)), 1.0, 0.0
                    ),
                ),
            ),
        )
    )

    markold = amr.udomarkOLD(u, lb, n=2)
    marknew = amr.udomark(u, lb, n=2)
    assert amr.jaccard(markold, marknew) == 1.0


if __name__ == "__main__":
    PARtest_refine_udo_parallelUDO()
    PARtest_udo_regression()
    PARtest_parallel_udo()
