import numpy as np
from firedrake import *
from viamr import VIAMR


def get_netgen_mesh(TriHeight=0.4):
    import netgen
    from netgen.geom2d import SplineGeometry
    width = 2
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


def test_viamr_spaces():
    mesh = get_netgen_mesh()
    V, W = VIAMR().spaces(mesh)
    assert V.dim() == 135
    assert W.dim() == 228


def test_viamr_mark_none():
    mesh = get_netgen_mesh()
    z = VIAMR()
    CG1, _ = z.spaces(mesh)
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    psi_ufl = conditional(le(r, r0), sqrt(
        1.0 - r * r), psi0 + dpsi0 * (r - r0))
    psi = Function(CG1).interpolate(psi_ufl)
    mark = z.udomark(mesh, psi, psi)  # all active
    assert norm(mark, 'L1') == 0.0
    mark = z.vcesmark(mesh, psi, psi)  # all active
    assert norm(mark, 'L1') == 0.0
    lift = Function(CG1).interpolate(psi + 1.0)
    mark = z.udomark(mesh, lift, psi)  # all inactive
    assert norm(mark, 'L1') == 0.0
    mark = z.vcesmark(mesh, lift, psi)  # all inactive
    assert norm(mark, 'L1') == 0.0


def test_overlapping_and_nonoverlapping_jaccard():
    mesh = get_netgen_mesh()
    z = VIAMR()
    _, DG0 = z.spaces(mesh)
    (x, y) = SpatialCoordinate(mesh)
    sol1active = Function(DG0).interpolate(
        conditional(x > 0, 1, 0))  # right half active
    sol2active = Function(DG0).interpolate(conditional(x > 0, 1, 0))  # overlap
    assert z.jaccard(sol1active, sol2active) == 1.0
    sol3active = Function(DG0).interpolate(
        conditional(x < -.5, 1, 0))  # smaller left half active
    assert z.jaccard(sol1active, sol3active) == 0.0


if __name__ == "__main__":
    test_netgen_mesh_creation()
    test_viamr_spaces()
    test_viamr_mark_none()
    test_overlapping_and_nonoverlapping_jaccard()
