import numpy as np
import pytest
from firedrake import *
from viamr import VIAMR
import subprocess
import os
import pathlib


def _get_netgen_mesh(TriHeight=0.4, width=2):
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


def _get_ball_obstacle(x, y):
    r = sqrt(x * x + y * y)
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def test_netgen_mesh_creation():
    mesh = _get_netgen_mesh()
    assert mesh.num_cells() == 228


def test_spaces():
    mesh = _get_netgen_mesh(TriHeight=1.2)
    CG1, DG0 = VIAMR(debug=True).spaces(mesh)
    assert CG1.dim() == 19
    assert DG0.dim() == 24


def test_mark_none():
    mesh = _get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
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
    discmark = Function(DG0).interpolate(_get_ball_obstacle(x, y) > 0.0)
    mark = amr.unionmarks(rightmark, discmark)
    # VTKFile(f"result_unionmarks.pvd").write(rightmark, discmark, mark)
    assert abs(assemble(mark * dx) - 9.92) < 1.0e-10  # union of marked area


def test_refine_udo():
    mesh = _get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 19
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
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


def test_refine_vcd():
    mesh = _get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 19
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
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
    mesh = _get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 19
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
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
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
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
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
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
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi + 1.0, psi))
    residual = -div(grad(u))  # largest near circle psi==0
    (imark, _, _) = amr.brinactivemark(u, psi, residual, theta=0.8)
    # VTKFile(f"result_brinactivemark.pvd").write(Function(CG1, name="diff").interpolate(u-psi), imark)
    rmesh = amr.refinemarkedelements(mesh, imark)
    # VTKFile(f"result_brinactivemark_refined.pvd").write(rmesh)
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 147


def test_overlapping_jaccard():
    mesh = _get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    x, _ = SpatialCoordinate(mesh)
    right = conditional(x > 0, 1, 0)
    active1 = Function(DG0).interpolate(right)  # right half active
    active2 = Function(DG0).interpolate(right)  # same; full overlap
    assert abs(amr.jaccard(active1, active2) - 1.0) < 1.0e-10


def test_nonoverlapping_jaccard():
    mesh = _get_netgen_mesh()
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    x, _ = SpatialCoordinate(mesh)
    right = conditional(x > 0, 1, 0)
    farleft = conditional(x < -0.5, 1, 0)
    active1 = Function(DG0).interpolate(right)
    active2 = Function(DG0).interpolate(farleft)  # no overlap
    assert abs(amr.jaccard(active1, active2) - 0.0) < 1.0e-10


def test_symmetry_jaccard():
    mesh = _get_netgen_mesh()
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
    test_petsc4py_refine_vcd()
    test_refine_vcd_petsc4py_firedrake()
