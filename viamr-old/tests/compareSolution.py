from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
from viamr import VIAMR
import subprocess

z = VIAMR()


subprocess.run(
    ["python3", "generateSolution.py", "--refinements", "2", "--runtime", "serial"]
)
subprocess.run(
    [
        "mpiexec",
        "-n",
        "4",
        "python3",
        "generateSolution.py",
        "--refinements",
        "2",
        "--runtime",
        "parallel",
    ]
)

with CheckpointFile("serialUDO.h5", "r") as afile:
    serialMesh = afile.load_mesh("serialMesh")
    serialMark = afile.load_function(serialMesh, "serialMark")


with CheckpointFile("parallelUDO.h5", "r") as afile:
    parallelMesh = afile.load_mesh("parallelMesh")
    parallelMark = afile.load_function(parallelMesh, "parallelMark")

print(z.jaccard(serialMark, parallelMark))
