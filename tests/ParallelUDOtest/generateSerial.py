# Testing Script for Parallel UDO Ideas
from firedrake import *
from firedrake.output import VTKFile

from viamr import VIAMR
from viamr import SpiralObstacleProblem


# Generate Mesh
initTriHeight = .1
problem_instance = SpiralObstacleProblem(TriHeight=initTriHeight)
mesh = problem_instance.setInitialMesh()
# Initialize VIAMR
z = VIAMR()

DG0 = FunctionSpace(mesh, "DG", 0)
testBorderElement = Function(DG0).interpolate(Constant(0.0))
testBorderElement.dat.data_wo_with_halos[[689, 848, 212]] = 1

mark = z.udomark(mesh, testBorderElement, n=3)
meshr = mesh.refine_marked_elements(mark)
mark.rename('mark')


VTKFile("resultsSerial.pvd").write(mark, testBorderElement)
with CheckpointFile("SerialUDO.h5", 'w') as afile:
    afile.save_mesh(mesh)
    afile.save_function(mark)
