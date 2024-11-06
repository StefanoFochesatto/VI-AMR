# Import necessary modules from Firedrake
from firedrake import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Read the Gmsh file
mesh = Mesh("Unif.msh", name="meshA")

(x, y) = SpatialCoordinate(mesh)
W = FunctionSpace(mesh, "DG", 0)
rank = Function(W, name='rank')
rank.dat.data[:] = mesh.comm.rank + 1
VTKFile("DistributedUnif.pvd").write(rank)
with CheckpointFile("DistributedUnif.h5", 'w') as afile:
    afile.save_mesh(mesh)
    afile.save_function(rank)
