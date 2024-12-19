# Import necessary modules from Firedrake
from firedrake import *
from viamr import VIAMR


# Read the Gmsh file
ExactMesh = Mesh(name="ExactMesh")

u = Function(W, name="u")

with CheckpointFile("ExactSolution.h5", 'w') as afile:
    afile.save_mesh(ExactMesh)
    afile.save_function(u)
