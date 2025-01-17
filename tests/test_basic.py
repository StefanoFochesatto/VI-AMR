import numpy as np
import pytest
from firedrake import *
from viamr import VIAMR


def get_netgen_mesh(TriHeight=0.4, width=2):
    import netgen
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-1 * width, -1 * width),
                     p2=(1 * width, 1 * width),
                     bc="rectangle")
    ngmsh = None
    ngmsh = geo.GenerateMesh(maxh=TriHeight)
    return Mesh(ngmsh)


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
    z = VIAMR(debug=True)
    CG1, _ = z.spaces(mesh)
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    mark = z.udomark(mesh, psi, psi)  # all active
    assert norm(mark, 'L1') == 0.0
    mark = z.vcesmark(mesh, psi, psi)  # all active
    assert norm(mark, 'L1') == 0.0
    lift = Function(CG1).interpolate(psi + 1.0)
    mark = z.udomark(mesh, lift, psi)  # all inactive
    assert norm(mark, 'L1') == 0.0
    mark = z.vcesmark(mesh, lift, psi)  # all inactive
    assert norm(mark, 'L1') == 0.0


def test_refine_udo():
    mesh = get_netgen_mesh(TriHeight=1.2)
    z = VIAMR(debug=True)
    CG1, _ = z.spaces(mesh)
    assert CG1.dim() == 19
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    #from firedrake.output import VTKFile
    #VTKFile(f"result_refine_0.pvd").write(u)
    mark = z.udomark(mesh, u, psi)
    rmesh = mesh.refine_marked_elements(mark)
    rCG1, _ = z.spaces(rmesh)
    assert rCG1.dim() == 61
    rV = FunctionSpace(rmesh, "CG", 1)
    ru = Function(rV).interpolate(u) # cross-mesh interpolation
    assert abs(norm(ru) - unorm0) < 1.0e-10 # ... should be conservative
    #VTKFile(f"result_refine_1.pvd").write(ru)


def test_refine_vces():
    mesh = get_netgen_mesh(TriHeight=1.2)
    z = VIAMR(debug=True)
    CG1, _ = z.spaces(mesh)
    assert CG1.dim() == 19
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    #from firedrake.output import VTKFile
    #VTKFile(f"result_refine_0.pvd").write(u)
    mark = z.vcesmark(mesh, u, psi)
    rmesh = mesh.refine_marked_elements(mark)
    rCG1, _ = z.spaces(rmesh)
    assert rCG1.dim() == 49
    rV = FunctionSpace(rmesh, "CG", 1)
    ru = Function(rV).interpolate(u) # cross-mesh interpolation
    assert abs(norm(ru) - unorm0) < 1.0e-10 # ... should be conservative
    #VTKFile(f"result_refine_1.pvd").write(ru)


def test_overlapping_jaccard():
    mesh = get_netgen_mesh(TriHeight=1.2)
    z = VIAMR(debug=True)
    _, DG0 = z.spaces(mesh)
    x, _ = SpatialCoordinate(mesh)
    right = conditional(x > 0, 1, 0)
    active1 = Function(DG0).interpolate(right)  # right half active
    active2 = Function(DG0).interpolate(right)  # same; full overlap
    assert z.jaccard(active1, active1) == 1.0


def test_nonoverlapping_jaccard():
    mesh = get_netgen_mesh()
    z = VIAMR(debug=True)
    _, DG0 = z.spaces(mesh)
    x, _ = SpatialCoordinate(mesh)
    right = conditional(x > 0, 1, 0)
    farleft = conditional(x < -.5, 1, 0)
    active1 = Function(DG0).interpolate(right)
    active2 = Function(DG0).interpolate(farleft)  # no overlap
    assert z.jaccard(active1, active2) == 0.0


def test_symmetry_jaccard():
    mesh = get_netgen_mesh()
    z = VIAMR(debug=True)
    _, DG0 = z.spaces(mesh)
    x, _ = SpatialCoordinate(mesh)
    right = conditional(x > 0, 1, 0)
    more = conditional(x < 1, 1, 0)
    active1 = Function(DG0).interpolate(right)
    active2 = Function(DG0).interpolate(more)
    assert z.jaccard(active1, active2) == z.jaccard(active2, active1)


def test_overlapping_and_nonoverlapping_hausdorff():
    # to have free boundaries line up with conditional statements
    mesh = RectangleMesh(10, 10, 1, 1)
    z = VIAMR()
    CG1, _ = z.spaces(mesh)
    x, y = SpatialCoordinate(mesh)
    sol1 = Function(CG1).interpolate(Constant(1.0))
    lb = conditional(x <= .2, 1, 0)
    _, E1 = z.freeboundarygraph(sol1, lb)
    assert z.hausdorff(E1, E1) == 0
    lb2 = conditional(x <= .4, 1, 0)
    _, E2 = z.freeboundarygraph(sol1, lb2)
    assert z.hausdorff(E1, E2) == .2


if __name__ == "__main__":
    test_netgen_mesh_creation()
    test_spaces()
    test_mark_none()
    test_refine_udo()
    test_refine_vces()
    test_overlapping_jaccard()
    test_nonoverlapping_jaccard()
    test_symmetry_jaccard()
    test_overlapping_and_nonoverlapping_hausdorff()
