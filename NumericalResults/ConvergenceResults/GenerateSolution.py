# Import necessary modules from Firedrake
from firedrake import *
from firedrake.output import VTKFile
from TestProblem import ObstacleProblem
from netgen.geom2d import CSG2d, EdgeInfo as EI, Solid2d


geo = CSG2d()

rect = Solid2d([
    (-2, -2),
    (-2, 2),
    (2, 2),
    (2, -2),
], mat="rect", bc="boundary")

r = 0.697965148223374
fbres = .0001
circle = Solid2d([
    (0, -r),
    EI((r,  -r), maxh=fbres),  # control point for quadratic spline
    (r, 0),
    EI((r,  r), maxh=fbres),  # spline with maxh
    (0, r),
    EI((-r,  r), maxh=fbres),
    (-r, 0),
    EI((-r, -r), maxh=fbres),  # spline with bc
])

geo.Add(circle)
geo.Add(rect)
ngmsh = geo.GenerateMesh(maxh=.001)
labels = [i+1 for i,
          name in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ["boundary"]]
ExactMesh = Mesh(ngmsh)

problem_instance = ObstacleProblem()
ExactU, lb, _ = problem_instance.solveProblem(
    mesh=ExactMesh, u=None, bdry=labels)
ExactU.rename("ExactU")

# Open issue won't run in parallel: https://github.com/firedrakeproject/firedrake/issues/3783
with CheckpointFile("ExactSolution.h5", 'w') as afile:
    afile.save_mesh(ExactMesh)
    afile.save_function(ExactU)