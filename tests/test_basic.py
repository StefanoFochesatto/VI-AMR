import numpy as np
import pytest
from firedrake import *
from viamr import VIAMR
import subprocess
import os
import pathlib
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



# for parallel testing
from mpi4py import MPI
from pytest_mpi.parallel_assert import parallel_assert


def get_netgen_mesh(TriHeight=0.4, width=2):
    import netgen
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle(
        p1=(-1 * width, -1 * width), p2=(1 * width, 1 * width), bc="rectangle"
    )
    ngmsh = None
    ngmsh = geo.GenerateMesh(maxh=TriHeight)
    return Mesh(
        ngmsh,
        distribution_parameters={
            "partition": True,
            "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
        },
    )


def test_netgen_mesh_creation():
    mesh = get_netgen_mesh()
    assert mesh.num_cells() == 228


def test_spaces():
    mesh = get_netgen_mesh(TriHeight=1.2)
    CG1, DG0 = VIAMR(debug=True).spaces(mesh)
    assert CG1.dim() == 19
    assert DG0.dim() == 24


def get_ball_obstacle(x, y):
    r = sqrt(x * x + y * y)
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def test_mark_none():
    mesh = get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    mark = amr.udomark(psi, psi)  # all active
    assert norm(mark, "L1") == 0.0
    mark = amr.vcdmark(psi, psi)  # all active
    assert norm(mark, "L1") == 0.0
    lift = Function(CG1).interpolate(psi + 1.0)
    mark = amr.udomark(lift, psi)  # all inactive
    assert norm(mark, "L1") == 0.0
    mark = amr.vcdmark(lift, psi)  # all inactive
    assert norm(mark, "L1") == 0.0


def test_unionmarks():
    mesh = RectangleMesh(
        5, 5, 2.0, 2.0, originX=-2.0, originY=-2.0
    )  # Firedrake utility mesh, not netgen
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    (x, y) = SpatialCoordinate(mesh)
    rightmark = Function(DG0).interpolate(conditional(x > 0.0, 1.0, 0.0))
    discmark = Function(DG0).interpolate(get_ball_obstacle(x, y) > 0.0)
    mark = amr.unionmarks(rightmark, discmark)
    # VTKFile(f"result_unionmarks.pvd").write(rightmark, discmark, mark)
    assert abs(assemble(mark * dx) - 9.92) < 1.0e-10  # union of marked area


def test_refine_udo():
    mesh = get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 19
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    # VTKFile(f"result_refine_0.pvd").write(u)
    mark = amr.udomark(u, psi)
    rmesh = mesh.refine_marked_elements(mark)  # netgen's refine method
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 61
    rV = FunctionSpace(rmesh, "CG", 1)
    ru = Function(rV).interpolate(u)  # cross-mesh interpolation
    assert abs(norm(ru) - unorm0) < 1.0e-10  # ... should be conservative
    # VTKFile(f"result_refine_1.pvd").write(ru)


def test_refine_udo_parallelUDO():
    mesh1 = get_netgen_mesh(TriHeight=0.1)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh1)
    (x, y) = SpatialCoordinate(mesh1)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    # VTKFile(f"result_refine_0.pvd").write(u)
    mark1 = amr.udomark(u, psi)
    rmesh1 = mesh1.refine_marked_elements(mark1)  # netgen's refine method
    mesh2 = get_netgen_mesh(TriHeight=0.1)
    CG1, _ = amr.spaces(mesh2)
    (x, y) = SpatialCoordinate(mesh1)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    # VTKFile("result_refine_0.pvd").write(u)
    mark2 = amr.udomark(u, psi)
    rmesh2 = mesh2.refine_marked_elements(mark2)  # netgen's refine method
    assert abs(amr.jaccard(mark1, mark2) - 1.0) < 1.0e-10
    r1CG1, _ = amr.spaces(rmesh1)
    r2CG1, _ = amr.spaces(rmesh2)
    assert r1CG1.dim() == r2CG1.dim()


def test_refine_vcd():
    mesh = get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 19
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    mark = amr.vcdmark(u, psi)
    rmesh = mesh.refine_marked_elements(mark)  # netgen's refine method
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 49
    rV = FunctionSpace(rmesh, "CG", 1)
    ru = Function(rV).interpolate(u)  # cross-mesh interpolation
    assert abs(norm(ru) - unorm0) < 1.0e-10  # ... should be conservative


def test_petsc4py_refine_vcd():
    mesh = get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 19
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    mark = amr.vcdmark(u, psi)
    rmesh = amr.refinemarkedelements(mesh, mark)
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 49
    rV = FunctionSpace(rmesh, "CG", 1)
    ru = Function(rV).interpolate(u)  # cross-mesh interpolation
    assert abs(norm(ru) - unorm0) < 1.0e-10  # ... should be conservative


def test_refine_vcd_petsc4py_firedrake():
    mesh = RectangleMesh(
        5, 5, 2.0, 2.0, originX=-2.0, originY=-2.0
    )  # Firedrake utility mesh, not netgen
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 36
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    mark = amr.vcdmark(u, psi)
    rmesh = amr.refinemarkedelements(mesh, mark)
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 73
    rV = FunctionSpace(rmesh, "CG", 1)
    ru = Function(rV).interpolate(u)  # cross-mesh interpolation
    assert abs(norm(ru) - unorm0) < 1.0e-10  # ... should be conservative


def test_gradrecinactivemark():
    mesh = RectangleMesh(6, 6, 2.0, 2.0, originX=-2.0, originY=-2.0, diagonal="crossed")
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 85
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi + 1.0, psi))
    imark, _ = amr.gradrecinactivemark(u, psi, theta=0.5)
    # VTKFile(f"result_gradrecinactivemark.pvd").write(Function(CG1, name="diff").interpolate(u-psi), imark)
    rmesh = amr.refinemarkedelements(mesh, imark)
    # VTKFile(f"result_gradrecinactivemark_refined.pvd").write(rmesh)
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 165


def test_brinactivemark():
    mesh = RectangleMesh(8, 8, 2.0, 2.0, originX=-2.0, originY=-2.0)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 81
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi + 1.0, psi))
    residual = -div(grad(u))  # largest near circle psi==0
    (imark, _, _) = amr.brinactivemark(u, psi, residual, theta=0.8)
    # VTKFile(f"result_brinactivemark.pvd").write(Function(CG1, name="diff").interpolate(u-psi), imark)
    rmesh = amr.refinemarkedelements(mesh, imark)
    # VTKFile(f"result_brinactivemark_refined.pvd").write(rmesh)
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 147


def test_overlapping_jaccard():
    mesh = get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    x, _ = SpatialCoordinate(mesh)
    right = conditional(x > 0, 1, 0)
    active1 = Function(DG0).interpolate(right)  # right half active
    active2 = Function(DG0).interpolate(right)  # same; full overlap
    assert abs(amr.jaccard(active1, active2) - 1.0) < 1.0e-10


def test_nonoverlapping_jaccard():
    mesh = get_netgen_mesh()
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    x, _ = SpatialCoordinate(mesh)
    right = conditional(x > 0, 1, 0)
    farleft = conditional(x < -0.5, 1, 0)
    active1 = Function(DG0).interpolate(right)
    active2 = Function(DG0).interpolate(farleft)  # no overlap
    assert abs(amr.jaccard(active1, active2) - 0.0) < 1.0e-10


def test_symmetry_jaccard():
    mesh = get_netgen_mesh()
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    x, _ = SpatialCoordinate(mesh)
    right = conditional(x > 0, 1, 0)
    more = conditional(x < 1, 1, 0)
    active1 = Function(DG0).interpolate(right)
    active2 = Function(DG0).interpolate(more)
    assert abs(amr.jaccard(active1, active2) - amr.jaccard(active2, active1)) < 1.0e-10


def test_overlapping_and_nonoverlapping_hausdorff():
    # to have free boundaries line up with conditional statements
    mesh = RectangleMesh(10, 10, 1, 1)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    x, y = SpatialCoordinate(mesh)
    sol1 = Function(CG1).interpolate(Constant(1.0))
    lb = Function(CG1).interpolate(conditional(x <= 0.2, 1, 0))
    _, E1 = amr.freeboundarygraph(sol1, lb)
    assert amr.hausdorff(E1, E1) == 0
    lb2 = Function(CG1).interpolate(conditional(x <= 0.4, 1, 0))
    _, E2 = amr.freeboundarygraph(sol1, lb2)
    assert amr.hausdorff(E1, E2) == 0.2
    
    
@pytest.mark.parallel(nprocs=3)
def test_parallel_udo():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    
    amr = VIAMR()
    mesh =  RectangleMesh(20, 20, 1, 1, distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)})
    CG1, _ = amr.spaces(mesh)
    u = Function(CG1).interpolate(1.0)
    x, y = SpatialCoordinate(mesh)
    
    # Somewhat interesting obstacle configuration. 
    lb = Function(CG1).interpolate(
       conditional(And(And(x > 0.15, x < 0.35), And(y > 0.15, y < 0.35)), 1.0,
        conditional(And(And(x > 0.65, x < 0.85), And(y > 0.15, y < 0.35)), 1.0,
        conditional(And(And(x > 0.15, x < 0.35), And(y > 0.65, y < 0.85)), 1.0,
        conditional(And(And(x > 0.65, x < 0.85), And(y > 0.65, y < 0.85)), 1.0, 0.0))))
    )
    
    # Mark in parallel
    mark = amr.udomark(u, lb, n = 2)
    
    # Compute number of local active elements
    localActive = np.sum(mark.dat.data_ro[:])
    globalActive = np.zeros(1, dtype=np.float64)
    comm.Allreduce(localActive, globalActive, op=MPI.SUM)
    
    if rank == 0:
        # Check agreement with serial implementation
        assert globalActive[0] == 506
    

def test_udo_regression():
    # This test utilizes the the old implementation of UDO which builds the neighborhood of the free boundary using breadth first search, 
    # as a regression test for the dmplex based implementation. 
    
    amr = VIAMRRegression()
    mesh =  RectangleMesh(20, 20, 1, 1, distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)})
    CG1, _ = amr.spaces(mesh)
    u = Function(CG1).interpolate(1.0)
    x, y = SpatialCoordinate(mesh)
    
    # Somewhat interesting obstacle configuration. 
    lb = Function(CG1).interpolate(
       conditional(And(And(x > 0.15, x < 0.35), And(y > 0.15, y < 0.35)), 1.0,
        conditional(And(And(x > 0.65, x < 0.85), And(y > 0.15, y < 0.35)), 1.0,
        conditional(And(And(x > 0.15, x < 0.35), And(y > 0.65, y < 0.85)), 1.0,
        conditional(And(And(x > 0.65, x < 0.85), And(y > 0.65, y < 0.85)), 1.0, 0.0))))
    )

    markold = amr.udomarkOLD(u, lb, n = 2)
    marknew = amr.udomark(u, lb, n = 2)
    assert amr.jaccard(markold, marknew) == 1.0
    


if __name__ == "__main__":
    test_netgen_mesh_creation()
    test_spaces()
    test_mark_none()
    test_unionmarks()
    test_refine_udo()
    test_refine_vcd()
    test_brinactivemark()
    test_overlapping_jaccard()
    test_nonoverlapping_jaccard()
    test_symmetry_jaccard()
    test_overlapping_and_nonoverlapping_hausdorff()
    test_refine_udo_parallelUDO()
    test_petsc4py_refine_vcd()
    test_refine_vcd_petsc4py_firedrake()
    test_udo_regression()
    test_parallel_udo()
