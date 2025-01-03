# Testing Script for Parallel UDO Ideas
from firedrake import *
from firedrake.output import VTKFile

from viamr import VIAMR
from viamr import SphereObstacleProblem

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Instantiate obstacle problem and solution
initTriHeight = .05
problem_instance = SphereObstacleProblem(TriHeight=initTriHeight)
u, lb, mesh = problem_instance.solveProblem(mesh=None, u=None)
dm = mesh.topology_dm

z = VIAMR()
nodalactive = z.nodalactive(u, lb)
elemborder = z.elemborder(nodalactive)

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
        if dm.getDepthStratum(2)[0] <= entity <= dm.getDepthStratum(2)[1]:
            NeighborSet.add(entity)

NeighborSet = list(NeighborSet)


# For plotting
_, DG0 = z.spaces(mesh)
Dilation = Function(DG0).interpolate(Constant(0.0))
plexelementlist = mesh.cell_closure[:, -1]
dm_to_fd = {number: index for index, number in enumerate(plexelementlist)}

for i in NeighborSet:
    Dilation.dat.data_wo_with_halos[dm_to_fd[i]] = 1

VTKFile("results.pvd").write(Dilation)
