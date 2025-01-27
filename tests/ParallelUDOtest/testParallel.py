from firedrake import *
from firedrake.output import VTKFile
from mpi4py import MPI
import numpy as np
from viamr import VIAMR
from viamr import SpiralObstacleProblem

try:
    import netgen
except ImportError:
    print("ImportError.  Unable to import NetGen.  Exiting.")
    import sys
    sys.exit(0)
from netgen.geom2d import SplineGeometry


# Define a new class that inherits from SpiralObstacleProblem
class CustomSpiralObstacleProblem(SpiralObstacleProblem):

    def setInitialMesh(self):
        geo = SplineGeometry()
        geo.AddRectangle(p1=(-1, -1),
                         p2=(1, 1), bc="rectangle")
        ngmsh = geo.GenerateMesh(maxh=self.TriHeight)
        mesh = Mesh(ngmsh, distribution_parameters={
            "partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)})
        return mesh


# Generate Mesh
initTriHeight = .1
problem_instance = CustomSpiralObstacleProblem(TriHeight=initTriHeight)
mesh = problem_instance.setInitialMesh()
# Initialize VIAMR
z = VIAMR()

# Setup indicator function
MPI.COMM_WORLD.Barrier()
DG0 = FunctionSpace(mesh, "DG", 0)
testBorderElement = Function(DG0).interpolate(Constant(0.0))
if MPI.COMM_WORLD.Get_size() > 1:
    testBorderElement.dat.data_wo_with_halos[232] = 1
else:
    testBorderElement.dat.data_wo_with_halos[[689, 848, 212]] = 1

# Writing out files breaks when n >= meshoverlapsize
# markp = z.udomarkParallel(mesh, testBorderElement, n=3)
meshp = mesh.refine_marked_elements(
    Function(FunctionSpace(mesh, "DG", 0)).interpolate(Constant(1.0)))
print(meshp._distribution_parameters)


VTKFile("resultsParallel.pvd").write(testBorderElement)
VTKFile("resultsParallelMarked.pvd").write(meshp)


with CheckpointFile("SerialUDO.h5", 'r') as afile:
    mesh = afile.load_mesh('Default')
    mark = afile.load_function(mesh, "mark")


# Calculate local sums
local_sum_parallel = sum(markp.dat.data)
local_sum_serial = sum(mark.dat.data)
MPI.COMM_WORLD.Barrier()
print(local_sum_parallel)
MPI.COMM_WORLD.Barrier()
# Initialize variables to store the reduced results on the root process
total_sum_parallel = np.zeros(1, dtype='d')
total_sum_serial = np.zeros(1, dtype='d')

# Reduce the local sums to the root process (rank 0)
MPI.COMM_WORLD.Reduce([local_sum_parallel, MPI.DOUBLE],
                      [total_sum_parallel, MPI.DOUBLE], op=MPI.SUM, root=0)
MPI.COMM_WORLD.Reduce([local_sum_serial, MPI.DOUBLE],
                      [total_sum_serial, MPI.DOUBLE], op=MPI.SUM, root=0)

# Print the total sums on rank 0
if MPI.COMM_WORLD.Get_rank() == 0:
    print(f"Parallel Marked Elements: {total_sum_parallel}")
    print(f"Serial Marked Elements: {total_sum_serial}")
