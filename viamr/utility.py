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
    def getInitialMesh(self):
        pass

    @abstractmethod
    def getObstacle(self, V, bdry=None):
        pass

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
            mesh = self.getInitialMesh()

        V = FunctionSpace(mesh, "CG", 1)

        if u is None:
            u = Function(V, name="u (FE soln)")
        else:
            u = Function(V, name="u (FE soln)").interpolate(u)

        lb, bcs, _ = self.getObstacle(V, bdry)

        v = TestFunction(V)
        F = inner(grad(u), grad(v)) * dx

        problem = NonlinearVariationalProblem(F, u, bcs)
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="")

        ub = Function(V).interpolate(Constant(PETSc.INFINITY))
        solver.solve(bounds=(lb, ub))

        return u, lb, mesh


class SphereObstacleProblem(BaseObstacleProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def getInitialMesh(self):
        geo = SplineGeometry()
        geo.AddRectangle(p1=(-2, -2),
                         p2=(2, 2), bc="rectangle")
        ngmsh = geo.GenerateMesh(maxh=self.TriHeight)
        mesh = Mesh(ngmsh)
        return mesh

    def getObstacleBC(self, V, bdry=None):
        """
        Sets up the obstacle problem by defining the obstacle function, sampled boundary condition function,
        and the Dirichlet boundary conditions object.

        Parameters:
        V     : FunctionSpace where the problem is defined.
        bdry  : Optional tuple specifying boundary indices. Defaults to all boundary indices (1, 2, 3, 4).

        Returns:
        psi   : Function representing the obstacle in the function space V.
        gbdry : Function object corresponding to the sampled boundary condition.
        bcs   : DirichletBC object that applies the boundary conditions to the problem.
        """

        mesh = V.mesh()
        (x, y) = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        r0 = 0.9
        psi0 = np.sqrt(1.0 - r0 * r0)
        dpsi0 = -r0 / psi0
        psi_ufl = conditional(le(r, r0), sqrt(
            1.0 - r * r), psi0 + dpsi0 * (r - r0))
        psi = Function(V).interpolate(psi_ufl)
        afree = 0.697965148223374
        A = 0.680259411891719
        B = 0.471519893402112
        gbdry_ufl = conditional(le(r, afree), psi_ufl, -A * ln(r) + B)
        gbdry = Function(V).interpolate(gbdry_ufl)

        if bdry is None:
            bdry = (1, 2, 3, 4)

        bcs = DirichletBC(V, gbdry, bdry)
        return psi, gbdry, bcs


# Example usage
# if __name__ == '__main__':
#    problem = SphereObstacleProblem()
#    solution, lower_bound, mesh = problem.solveProblem()
#    print("Solution obtained successfully.")
