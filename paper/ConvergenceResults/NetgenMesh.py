# Import necessary modules from Firedrake
from firedrake import *
from firedrake.output import VTKFile
from paper.convergence.utility import SphereObstacleProblem
from netgen.geom2d import CSG2d, EdgeInfo as EI, Solid2d
from viamr import VIAMR
import netgen.gui


geo = CSG2d()

rect = Solid2d([
    (-2, -2),
    (-2, 2),
    (2, 2),
    (2, -2),
], mat="rect", bc="boundary")

r = 0.697965148223374
fbres = .001
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
ngmsh = geo.GenerateMesh(maxh=.01)
labels = [i+1 for i,
          name in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ["boundary"]]


ExactMesh = Mesh(ngmsh)
# Run in debugger with import netgen.gui and export the mesh. Then open in gmsh, remove the 2d surface and save as .msh file. 
print("test")