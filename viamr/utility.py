from abc import ABC, abstractmethod
import numpy as np

from firedrake import *
from firedrake.petsc import OptionsManager, PETSc
from firedrake.output import VTKFile


from viamr import VIAMR

try:
    import netgen
except ImportError:
    print("ImportError.  Unable to import NetGen.  Exiting.")
    import sys
    sys.exit(0)
from netgen.geom2d import SplineGeometry


class BaseObstacleProblem(ABC, OptionsManager):
    def __init__(self, **kwargs):
        self.TriHeight = kwargs.pop("TriHeight", 0.1)
        sp_default = {  
          # "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-12,
            "snes_stol": 1.0e-12,
            "snes_type": "vinewtonrsls",
            "snes_vi_zero_tolerance": 1.0e-12,
            "snes_linesearch_type": "basic",
            "snes_max_it": 200,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"}
        self.sp = kwargs.pop("sp", sp_default)
        
        sp_fascd = {
            "fascd_cycle_type": "full",
            "fascd_skip_downsmooth": None, 
            "fascd_monitor": None,
            "fascd_converged_reason": None,
            "fascd_levels_snes_type": "vinewtonrsls",
            "fascd_levels_snes_max_it": 1,
            "fascd_levels_snes_converged_reason": None,
            "fascd_levels_snes_vi_zero_tolerance": 1.0e-12,
            "fascd_levels_snes_linesearch_type": "basic",
            "fascd_levels_ksp_type": "cg",
            "fascd_levels_ksp_max_it": 3,
            "fascd_levels_ksp_converged_maxits": None,
            "fascd_levels_pc_type": "bjacobi",
            "fascd_levels_sub_pc_type": "icc",
            "fascd_coarse_snes_converged_reason": None,
            "fascd_coarse_snes_rtol": 1.0e-8,   
            "fascd_coarse_snes_atol": 1.0e-12,
            "fascd_coarse_snes_stol": 1.0e-12,
            "fascd_coarse_snes_type": "vinewtonrsls",
            "fascd_coarse_snes_vi_zero_tolerance": 1.0e-12,
            "fascd_coarse_snes_linesearch_type": "basic",
            "fascd_coarse_ksp_type": "preonly",
            "fascd_coarse_pc_type": "lu",
            "fascd_coarse_pc_factor_mat_solver_type": "mumps"}
        self.spFASCD = kwargs.pop("spFASCD", sp_fascd)


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
        lb = Function(V).interpolate(obsUFL)
        lb.rename("lb (lower bound obstacle)")
        return lb

    def getBoundaryConditions(self, V):
        bdryUFL, bdryID = self.setBoundaryConditionsUFL(V)
        bdry = Function(V).interpolate(bdryUFL)
        bcs = DirichletBC(V, bdry, bdryID)
        return bcs, bdry

    def getExactSolution(self, V):
        return None

    def solveProblem(self, mesh=None, u=None, bdry=None, moreparams=None, FASCD = False):
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
        ub = Function(V).interpolate(Constant(PETSc.INFINITY))
        
        if FASCD:
            from fascd import FASCDSolver
            params = self.spFASCD.copy()
            if moreparams is not None:
                params.update(moreparams)
                
            solver = FASCDSolver(problem, solver_parameters=params, options_prefix="", bounds=(lb, ub),
                                     debug_coarse_save=False, warnings_convergence=True,
                                     admiss_fail_dump=True)
            solver.solve()
            
            
        else:
            params = self.sp.copy()
            if moreparams is not None:
                params.update(moreparams)
                
            solver = NonlinearVariationalSolver(
                problem, solver_parameters=params, options_prefix=""
            )
            solver.solve(bounds=(lb, ub))
        
        return u, lb


class SphereObstacleProblem(BaseObstacleProblem):
    """
    Example for constructing an obstacle problem using abstract base classes to simplify usage.
    All that is needed is to specify the initial mesh, obstacle UFL, and boundary conditions UFL expressions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setInitialMesh(self):
        geo = SplineGeometry()
        geo.AddRectangle(p1=(-2, -2), p2=(2, 2), bc="rectangle")
        ngmsh = geo.GenerateMesh(maxh=self.TriHeight)
        mesh = Mesh(ngmsh, distribution_parameters={
                    "partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)})
        return mesh

    def setObstacleUFL(self, V):
        mesh = V.mesh()
        (x, y) = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        r0 = 0.9
        psi0 = np.sqrt(1.0 - r0 * r0)
        dpsi0 = -r0 / psi0
        obsUFL = conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))
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

    def getExactSolution(self, V):
        gbdryUFL, _ = self.setBoundaryConditionsUFL(V)
        return Function(V).interpolate(gbdryUFL)


class SpiralObstacleProblem(BaseObstacleProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setInitialMesh(self):
        geo = SplineGeometry()
        geo.AddRectangle(p1=(-1, -1), p2=(1, 1), bc="rectangle")
        ngmsh = geo.GenerateMesh(maxh=self.TriHeight)
        mesh = Mesh(ngmsh, distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)})
        return mesh

    def setObstacleUFL(self, V):
        mesh = V.mesh()
        (x, y) = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        theta = atan2(y, x)
        tmp = (
            sin(2.0 * pi / r + pi / 2.0 - theta)
            + r * (r + 1) / (r - 2.0)
            - 3.0 * r
            + 3.6
        )
        obsUFL = conditional(le(r, 1.0e-8), 3.6, tmp)
        return obsUFL

    def setBoundaryConditionsUFL(self, V, bdryID=None):
        bdryUFL = Constant(0.0)
        if bdryID is None:
            bdryID = (1, 2, 3, 4)
        return bdryUFL, bdryID


class LShapedDomainProblem(BaseObstacleProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setInitialMesh(self, filename = None):
        if filename == None:
            # L shaped domain, missing 4th quadrant
            geo = SplineGeometry()
            interiorX = 1.25
            interiorY = -1.25
            pnts = [(interiorX, interiorY), (6, interiorY), (6, 2), (2, 2), (-2, 2), (-2, -2), (-2, -6), (interiorX, -6)]
            p1, p2, p3, p4, p5, p6, p7, p8 = [geo.AppendPoint(*pnt) for pnt in pnts]
            curves = [
                [["line", p1, p2], "line"],
                [["line", p2, p3], "line"],
                [["line", p3, p4], "line"],
                [["line", p4, p5], "line"],
                [["line", p5, p6], "line"],
                [["line", p6, p7], "line"],
                [["line", p7, p8], "line"],
                [["line", p8, p1], "line"],
            ]
            [geo.Append(c, bc=bc) for c, bc in curves]
            ngmsh = geo.GenerateMesh(maxh=self.TriHeight)
            mesh = Mesh(ngmsh, distribution_parameters={
                        "partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)})
        else:
            mesh = Mesh(filename, distribution_parameters={
                        "partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)})

        return mesh

    def setObstacleUFL(self, V):
        mesh = V.mesh()
        (x, y) = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        r0 = 0.9
        psi0 = np.sqrt(1.0 - r0 * r0)
        dpsi0 = -r0 / psi0
        obsUFL = conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))
        return obsUFL
        # Bump function parameters
        #center = [-0.5, 0.5]
        #width = (0.5,)
        #height = 0.25

        #mesh = V.mesh()
        #(x, y) = SpatialCoordinate(mesh)
        #dsqr = (x + 0.5) ** 2 + (y - 0.5) ** 2
        #obsUFL = (1.5 * exp(-2.5 * dsqr)) - 1
        #return obsUFL

    def setBoundaryConditionsUFL(self, V, bdryID=None):
        mesh = V.mesh()
        (x, y) = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        afree = 0.697965148223374
        A = 0.680259411891719
        B = 0.471519893402112

        obsUFL = self.setObstacleUFL(V)
        #bdryUFL = conditional(le(r, afree), obsUFL, -A * ln(r) + B)
        bdryUFL = Constant(0.0)
        #if V.mesh().netgen_mesh is not None:
        #    ngmsh = V.mesh().netgen_mesh
        #    bdryID = [
        #        i + 1
        #        for i, name in enumerate(ngmsh.GetRegionNames(codim=1))
        #        if name in ["line"]
        #        ]
    
        if bdryID is None:
            bdryID = (1, 2, 3, 4, 5, 6, 7, 8)

        return bdryUFL, bdryID    
        
        #bdryUFL = Constant(0.0)
        # Pull ngmsh from V.mesh() and get the boundary IDs for the boundary conditions
        #ngmsh = V.mesh().netgen_mesh
        #bdryID = [
        #    i + 1
        #    for i, name in enumerate(ngmsh.GetRegionNames(codim=1))
        #    if name in ["line"]
        #]
        #if bdryID is None:
        #    bdryID = (1, 2, 3, 4)
        #return bdryUFL, bdryID


# Example usage
#if __name__ == '__main__':
#    u = None
#    problem = LShapedDomainProblem(TriHeight=.05)
#    amr = VIAMR()
#    mesh = problem.setInitialMesh()
#    meshHist = [mesh]
#    for i in range(3):
#        u, lb = problem.solveProblem(mesh=meshHist[i], u=u)
#        mark = amr.udomark(meshHist[i], u, lb, n=1)
#        mesh = meshHist[i].refine_marked_elements(mark)
#        meshHist.append(mesh)
#    VTKFile('lshape.pvd').write(u)
