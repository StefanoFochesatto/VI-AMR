# Testing Script for Parallel UDO Ideas
from firedrake import *
from firedrake.output import VTKFile

from viamr import VIAMR
from viamr import SphereObstacleProblem

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 3

activetol = 1.0e-10
# Instantiate obstacle problem and solution
initTriHeight = .05
problem_instance = SphereObstacleProblem(TriHeight=initTriHeight)
mesh = problem_instance.setInitialMesh()
dm = mesh.topology_dm
mesh = Mesh(dm, distribution_parameters={
            "partition": None, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)})


u, lb = problem_instance.solveProblem(mesh=mesh, u=None)
# Entrypoint to udomark

DG0 = FunctionSpace(mesh, "DG", 0)
CG1 = FunctionSpace(mesh, "CG", 1)

nodalactive = Function(CG1).interpolate(
    conditional(abs(u - lb) < activetol, 0, 1))
elemborder = Function(DG0).interpolate(conditional(
    nodalactive > 0, conditional(nodalactive < 1, 1, 0), 0))

for i in range(n):
    # Pull border elements cell with dmplex cell indices
    BorderSetElementsIndices = [mesh.cell_closure[:, -1][i] for i, value in enumerate(
        elemborder.dat.data_ro_with_halos) if value != 0]

    # Pull indices of vertices which are incident to said border elements
    incidentVertices = [dm.getTransitiveClosure(
        i)[0][4:7] for i in BorderSetElementsIndices]

    # Flatten the list of lists and remove duplicates
    flat_list = [vertex for sublist in incidentVertices for vertex in sublist]
    incidentVertices = list(set(flat_list))

    # Pull all elements which are neighbor to the incidentVertices. This produces the set N(B)
    NeighborSet = set()
    for i in incidentVertices:
        for entity in dm.getTransitiveClosure(i, useCone=False)[0]:
            if dm.getDepthStratum(2)[0] <= entity < dm.getDepthStratum(2)[1]:
                NeighborSet.add(entity)

    NeighborSet = list(NeighborSet)

    # For plotting
    elemborder = Function(DG0).interpolate(Constant(0.0))
    plexelementlist = mesh.cell_closure[:, -1]

    dm_to_fd = {number: index for index, number in enumerate(plexelementlist)}

    for i in NeighborSet:
        elemborder.dat.data_wo_with_halos[dm_to_fd[i]] = 1

    print("flag")

    # Synchronize all processes
    MPI.COMM_WORLD.Barrier()


VTKFile("results.pvd").write(elemborder)
