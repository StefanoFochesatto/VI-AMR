import argparse
from firedrake import *
from firedrake.output import VTKFile

from viamr import VIAMR
from viamr import SpiralObstacleProblem
from firedrake.petsc import PETSc


# python3 generateSolution.py --refinements 2 --runtime serial
# or in parallel
# mpiexec -n 4 python3 generateSolution.py --refinements 2 --runtime parallel

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Script for Parallel UDO Ideas Testing')
parser.add_argument('--refinements', type=int, default=3,
                    help='Number of refinement levels (default: 1)')
parser.add_argument('--runtime', type=str, default='serial',
                    choices=['serial', 'parallel'],
                    help='Runtime type (options: serial, parallel)')
args = parser.parse_args()

# Generate Mesh
initTriHeight = 0.1
problem_instance = SpiralObstacleProblem(TriHeight=initTriHeight)
mesh = problem_instance.setInitialMesh()
meshHist = [mesh]
u = None

# Initialize VIAMR
z = VIAMR()

for i in range(args.refinements):
    PETSc.Sys.Print("solving problem")
    u, lb = problem_instance.solveProblem(mesh, u)
    PETSc.Sys.Print("problem solved")

    PETSc.Sys.Print("UDO marking")
    mark = z.udomarkParallel(mesh, u, lb, n=1)
    PETSc.Sys.Print("UDO marked")
    if i < args.refinements - 1:
        PETSc.Sys.Print("refining")
        mesh = z.refinemarkedelements(mesh, mark)
        PETSc.Sys.Print("refined")

if args.refinements > 0:
    mesh.name = f"{args.runtime}Mesh"
    mark.rename(f"{args.runtime}Mark")

VTKFile(f"results{args.runtime}.pvd").write(mark)

with CheckpointFile(f"{args.runtime}UDO.h5", 'w') as afile:
    afile.save_mesh(mesh)
    afile.save_function(mark)
