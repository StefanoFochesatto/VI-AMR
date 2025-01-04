from abc import ABC, abstractmethod
from netgen.geom2d import SplineGeometry
from firedrake import *
from firedrake.petsc import OptionsManager, PETSc
from firedrake.output import VTKFile
import numpy as np


class BaseObstacleProblem(ABC, OptionsManager):
    def __init__(self, **kwargs):
        self.activetol = kwargs.pop("activetol", 1.0e-10)
        self.TriHeight = kwargs.pop("TriHeight", 0.45)
        self.debug = kwargs.pop('debug', False)

    @abstractmethod
    def setInitialMesh(self):
        pass

    @abstractmethod
    def setBoundaryConditionsUFL(self, V, bdryID=None):
        pass

    @abstractmethod
    def setObstacleUFL(self, V):
        pass

    def getObstacle(self, V):
        obsUFL = self.setObstacleUFL(V)

        return Function(V).interpolate(obsUFL)

    def getBoundaryConditions(self, V):
        bdryUFL, bdryID = self.setBoundaryConditionsUFL(V)
        bdry = Function(V).interpolate(bdryUFL)
        bcs = DirichletBC(V, bdry, bdryID)

        return bcs, bdry

    def solveProblem(self, mesh=None, u=None, bdry=None,  sp={
        "snes_vi_monitor": None,
        "snes_type": "vinewtonrsls",
        "snes_converged_reason": None,
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-12,
        "snes_stol": 1.0e-12,
        "snes_max_it": 10000,
        "snes_vi_zero_tolerance": 1.0e-12,
        "snes_linesearch_type": "basic",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }):
        if mesh is None:
            mesh = self.setInitialMesh()

        V = FunctionSpace(mesh, "CG", 1)

        if u is None:
            u = Function(V, name="u (FE soln)")
        else:
            u = Function(V, name="u (FE soln)").interpolate(u)

        lb = self.getObstacle(V)
        bcs, _ = self.getBoundaryConditions(V)

        v = TestFunction(V)
        F = inner(grad(u), grad(v)) * dx

        problem = NonlinearVariationalProblem(F, u, bcs)
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="")

        ub = Function(V).interpolate(Constant(PETSc.INFINITY))
        solver.solve(bounds=(lb, ub))

        return u, lb, mesh


class SphereObstacleProblem(BaseObstacleProblem):
    '''
    Example for constructing an obstacle problem using abstract base classes to simplify usage.
    All that is needed is to specify the initial mesh, obstacle UFL, and boundary conditions UFL expressions. 
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setInitialMesh(self):
        geo = SplineGeometry()
        geo.AddRectangle(p1=(-2, -2),
                         p2=(2, 2), bc="rectangle")
        ngmsh = geo.GenerateMesh(maxh=self.TriHeight)
        mesh = Mesh(ngmsh)
        return mesh

    def setObstacleUFL(self, V):
        mesh = V.mesh()
        (x, y) = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        r0 = 0.9
        psi0 = np.sqrt(1.0 - r0 * r0)
        dpsi0 = -r0 / psi0
        obsUFL = conditional(le(r, r0), sqrt(
            1.0 - r * r), psi0 + dpsi0 * (r - r0))
        return obsUFL

    def setBoundaryConditionsUFL(self, V, bdryID=None):
        mesh = V.mesh()
        (x, y) = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        afree = 0.697965148223374
        A = 0.680259411891719
        B = 0.471519893402112

        obsUFL = self.setObstacleUFL(V)
        bdryUFL = conditional(le(r, afree), obsUFL, -A * ln(r) + B)

        if bdryID is None:
            bdryID = (1, 2, 3, 4)

        return bdryUFL, bdryID


class SpiralObstacleProblem(BaseObstacleProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setInitialMesh(self):
        geo = SplineGeometry()
        geo.AddRectangle(p1=(-1, -1),
                         p2=(1, 1), bc="rectangle")
        ngmsh = geo.GenerateMesh(maxh=self.TriHeight)
        mesh = Mesh(ngmsh)
        return mesh

    def setObstacleUFL(self, V):
        mesh = V.mesh()
        (x, y) = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        theta = atan2(y, x)
        tmp = sin(2.0*pi/r + pi/2.0 - theta) + r * \
            (r+1) / (r - 2.0) - 3.0 * r + 3.6
        obsUFL = conditional(le(r, 1.0e-8), 3.6, tmp)
        return obsUFL

    def setBoundaryConditionsUFL(self, V, bdryID=None):
        bdryUFL = Constant(0.0)
        if bdryID is None:
            bdryID = (1, 2, 3, 4)

        return bdryUFL, bdryID


# # Example usage
# if __name__ == '__main__':
#     problem = SpiralObstacleProblem(TriHeight=.05)  # Instantiate your problem
#     # Pass initial iterate, refined mesh, or solver parameter dictionary.
#     solution, lower_bound, mesh = problem.solveProblem()
#     print("Solution obtained successfully.")
