from firedrake import *
from firedrake.output import VTKFile
from mpi4py import MPI

from viamr import VIAMR
from viamr import SphereObstacleProblem


initTriHeight = .05
problem_instance = SphereObstacleProblem(TriHeight=initTriHeight)
u, lb, meshp = problem_instance.solveProblem(mesh=None, u=None)
z = VIAMR()
markp = z.udomarkParallel(meshp, u, lb, n=3)
markp.rename('markp')

with CheckpointFile("SerialUDO.h5", 'r') as afile:
    mesh = afile.load_mesh('Default')
    mark = afile.load_function(mesh, "mark")


# Calculate local sums
local_sum_parallel = sum(markp.dat.data)
local_sum_serial = sum(mark.dat.data)

# Reduce sums to rank 0
total_sum_parallel = MPI.COMM_WORLD.reduce(
    local_sum_parallel, op=MPI.SUM, root=0)
total_sum_serial = MPI.COMM_WORLD.reduce(local_sum_serial, op=MPI.SUM, root=0)

# Print the total sums on rank 0
if MPI.COMM_WORLD.Get_rank() == 0:
    print(f"Parallel Marked Elements: {total_sum_parallel}")
    print(f"Serial Marked Elements: {total_sum_serial}")
