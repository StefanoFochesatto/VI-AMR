import numpy as np
import firedrake as fd


def readnc(filename, vname, preview=False):
    '''read data from variable with name vname into 2d numpy array
    assumes fixed dimension variable names 'x1', 'y1' '''
    import netCDF4
    data = netCDF4.Dataset(filename)
    data.set_auto_mask(False)  # otherwise irritating masked arrays
    v = data.variables[vname][0,:,:].T  # transpose immediately
    x = data.variables['x1']
    y = data.variables['y1']
    if preview:
        import matplotlib.pyplot as plt
        plt.pcolormesh(x, y, v)
        plt.axis('equal')
        plt.show()
    return x, y, v


def _corners(x, y):
    ll = (min(x), min(y))  # lower left
    ur = (max(x), max(y))  # upper right
    return ll, ur


def fnmesh(m, x, y):
    '''generate a Firedrake/NetGen mesh for rectangle defined by
    numpy arrays x,y.  Resulting mesh has approximately m elements
    in the shorter dimension.'''
    llxy, urxy = _corners(x, y)
    try:
        import netgen
    except ImportError:
        printpar("ImportError.  Unable to import NetGen.  Exiting.")
        import sys
        sys.exit(0)
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle(p1=llxy, p2=urxy, bc="rectangle")
    trih = max(urxy[0] - llxy[0], urxy[1] - llxy[1]) / m
    ngmsh = geo.GenerateMesh(maxh=trih)
    mesh = fd.Mesh(ngmsh)
    return mesh


def datamesh(x, y):
    '''create rectangular Firedrake data mesh from numpy arrays x,y'''
    llxy, urxy = _corners(x, y)
    mx, my = len(x), len(y)
    dmesh = fd.RectangleMesh(mx - 1, my - 1,
                             urxy[0] - llxy[0], urxy[1] - llxy[1])
    dmesh.coordinates.dat.data[:,0] += llxy[0]
    dmesh.coordinates.dat.data[:,1] += llxy[1]
    return dmesh


def field_on_datamesh(dmesh, x, y, f, delnear=100.0e3):
    '''read 2d numpy array f into a Firedrake CG1 function on dmesh;
    also returns a Firedrake CG1 function which is 1 near the boundary
    and zero otherwise'''
    llxy, urxy = _corners(x, y)
    hx, hy = x[1] - x[0], y[1] - y[0]
    dCG1 = fd.FunctionSpace(dmesh, "CG", 1)
    fCG1 = fd.Function(dCG1)
    nearCG1 = fd.Function(dCG1)  # set to zero here
    for k in range(len(fCG1.dat.data)):
        xk, yk = dmesh.coordinates.dat.data[k]
        i = int((xk - llxy[0]) / hx)
        j = int((yk - llxy[1]) / hy)
        fCG1.dat.data[k] = f[i][j]
        db = min([abs(xk - llxy[0]), abs(xk - urxy[0]), abs(yk - llxy[1]), abs(yk - urxy[1])])
        if db < delnear:
            nearCG1.dat.data[k] = 1.0
    return fCG1, nearCG1
