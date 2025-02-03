import numpy as np
import pytest
from firedrake import *
from viamr import VIAMR
import subprocess

def get_netgen_mesh(TriHeight=0.4, width=2):
    import netgen
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-1 * width, -1 * width),
                     p2=(1 * width, 1 * width),
                     bc="rectangle")
    ngmsh = None
    ngmsh = geo.GenerateMesh(maxh=TriHeight)
    return Mesh(ngmsh, distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)})


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
    
    
    
    
def test_refine_udo_parallelUDO():
    mesh1 = get_netgen_mesh(TriHeight=.1)
    z = VIAMR(debug=True)
    CG1, _ = z.spaces(mesh1)
    (x, y) = SpatialCoordinate(mesh1)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    #from firedrake.output import VTKFile
    #VTKFile(f"result_refine_0.pvd").write(u)
    mark1 = z.udomark(mesh1, u, psi)
    rmesh1 = mesh1.refine_marked_elements(mark1)
    
    mesh2 = get_netgen_mesh(TriHeight=.1)
    CG1, _ = z.spaces(mesh2)
    (x, y) = SpatialCoordinate(mesh1)
    psi = Function(CG1).interpolate(get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    # from firedrake.output import VTKFile
    # VTKFile(f"result_refine_0.pvd").write(u)
    mark2 = z.udomarkParallel(mesh1, u, psi)
    rmesh2 = mesh2.refine_marked_elements(mark2)
    
    assert z.jaccard(mark1, mark2) == 1.0
    
    
    r1CG1, _ = z.spaces(rmesh1)
    r2CG1, _ = z.spaces(rmesh2)

    assert r1CG1.dim() == r2CG1.dim()
    

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
    #VTKFile(f"netgen_result_refine_1.pvd").write(ru)
    
    


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
    
def test_parallel_udo():
    # This test is not well encapsulated at all however barring crazy changes to the spiral utility problem
    # and jaccard, we have good visibility of parallel udo and refinemarkedelements()
     
    # Run UDO method in both parallel and serial
    z = VIAMR()
    subprocess.run(["python3", "generateSolution.py",
                "--refinements", "2", "--runtime", "serial"])
    subprocess.run(
        ["mpiexec", "-n", "4", "python3", "generateSolution.py",
        "--refinements", "2", "--runtime", "parallel"])

    # Checkpoint in files
    with CheckpointFile("serialUDO.h5", 'r') as afile:
        serialMesh = afile.load_mesh("serialMesh")
        serialMark = afile.load_function(serialMesh, "serialMark")


    with CheckpointFile("parallelUDO.h5", 'r') as afile:
        parallelMesh = afile.load_mesh("parallelMesh")
        parallelMark = afile.load_function(parallelMesh, "parallelMark")

    #Compare overlap, perfect overlap will have jaccard 1. 
    assert z.jaccard(serialMark, parallelMark) == 1.0



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
    test_refine_udo_parallelUDO()
    test_parallel_udo()
