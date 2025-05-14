import numpy as np
import firedrake as fd

class DataNetCDF():   # FIXME should it subclass something?

    def __init__(self, filename, vname, xname='x1', yname='y1'):
        '''constructor opens NetCDF4 file (filename) and reads variable (varname)
        into 2d numpy array; this defines a rectangular structured grid'''
        import netCDF4
        data = netCDF4.Dataset(filename)
        data.set_auto_mask(False)  # otherwise irritating masked arrays
        self.vname = vname
        self.v = data.variables[vname][0,:,:].T  # transpose immediately
        self.x = data.variables[xname]
        self.y = data.variables[yname]
        self.mx, self.my = np.shape(self.v)
        assert self.mx == len(self.x)
        assert self.my == len(self.y)
        self.ll = (min(self.x), min(self.y))  # lower left
        self.ur = (max(self.x), max(self.y))  # upper right
        self.Lx, self.Ly = self.ur[0] - self.ll[0], self.ur[1] - self.ll[1]
        self.hx, self.hy = self.x[1] - self.x[0],   self.y[1] - self.y[0]


    def describe_grid(self, print=print, indent=4):
        indentstr = indent * ' '
        llstr = f'({self.ll[0]/1000.0:.3f},{self.ll[1]/1000.0:.3f})'
        urstr = f'({self.ur[0]/1000.0:.3f},{self.ur[1]/1000.0:.3f})'
        print(f'{indentstr}rectangle {llstr}-->{urstr} km')
        print(f'{indentstr}{self.mx} x {self.my} grid with {self.hx/1000.0:.3f} x {self.hy/1000.0:.3f} km spacing')


    def preview(self):
        import matplotlib.pyplot as plt
        plt.pcolormesh(self.x, self.y, self.v.T, shading='nearest')
        plt.axis('equal')
        plt.title(f'{self.vname} (CLOSE FIGURE TO CONTINUE)')
        plt.show()


    def ngmesh(self, m):
        '''generate a Firedrake/NetGen mesh for the rectangle, with
        approximately m elements in the shorter dimension.'''
        try:
            import netgen
        except ImportError:
            printpar("ImportError.  Unable to import NetGen.  Exiting.")
            import sys
            sys.exit(0)
        from netgen.geom2d import SplineGeometry
        geo = SplineGeometry()
        geo.AddRectangle(p1=self.ll, p2=self.ur, bc="rectangle")
        trih = max(self.Lx, self.Ly) / m
        ngmsh = geo.GenerateMesh(maxh=trih)
        mesh = fd.Mesh(ngmsh)
        return mesh


    def function(self, delnear=100.0e3):
        '''return a Firedrake CG1 function on a rectangular P1 (CG1 triangles)
        Firedrake data mesh matching the vertices read from NetCDF file;
        also returns a Firedrake CG1 function which is 1 near the boundary
        and zero otherwise; recover the mesh itself by VAR.function_space().mesh()'''
        dmesh = fd.RectangleMesh(len(self.x) - 1, len(self.y) - 1, self.Lx, self.Ly)
        dmesh.coordinates.dat.data[:,0] += self.ll[0]
        dmesh.coordinates.dat.data[:,1] += self.ll[1]
        dCG1 = fd.FunctionSpace(dmesh, "CG", 1)
        fCG1 = fd.Function(dCG1)
        nearCG1 = fd.Function(dCG1)  # set to zero here
        for k in range(len(fCG1.dat.data)):
            xk, yk = dmesh.coordinates.dat.data[k]
            i = int((xk - self.ll[0]) / self.hx)
            j = int((yk - self.ll[1]) / self.hy)
            fCG1.dat.data[k] = self.v[i][j]
            db = min([abs(xk - self.ll[0]), abs(xk - self.ur[0]), abs(yk - self.ll[1]), abs(yk - self.ur[1])])
            if db < delnear:
                nearCG1.dat.data[k] = 1.0
        return fCG1, nearCG1
