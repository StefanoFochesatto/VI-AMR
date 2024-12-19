# Import necessary modules from Firedrake
from firedrake import *
from firedrake.output import VTKFile
from netgen.geom2d import CSG2d, EdgeInfo as EI, Solid2d


geo = CSG2d()

# define some primitives

rect = Solid2d([
    (-2, -2),
    (-2, 2),
    (2, 2),
    (2, -2),
], mat="rect")

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
ngmsh = geo.GenerateMesh(maxh=.2)
mesh = Mesh(ngmsh)
VTKFile("NetgenTest.pvd").write(mesh)
