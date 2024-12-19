# Import Firedrake and Netgen
from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR


class ObstacleProblem(OptionsManager):

    def __init__(self, **kwargs):
        self.activetol = kwargs.pop("activetol", 1.0e-10)
        self.TriHeight = kwargs.pop("TriHeight", .45)
        self.problem = kwargs.pop("Problem", "sphere")
        self.debug = kwargs.pop('debug', False)
        if self.problem == "sphere":
            self.width = 2
        else:
            self.width = 1

    def getInitialMesh(self):
        geo = SplineGeometry()
        geo.AddRectangle(p1=(-1*self.width, -1*self.width),
                         p2=(1*self.width, 1*self.width),
                         bc="rectangle")

        ngmsh = geo.GenerateMesh(maxh=self.TriHeight)
        mesh = Mesh(ngmsh)
        mesh.topology_dm.viewFromOptions('-dm_view')

        return mesh

    def sphere_problem(mesh, V):
        # exactly-solvable obstacle problem for spherical obstacle
        # see Chapter 12 of Bueler (2021)
        # obstacle and solution are in function space V
        (x, y) = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        r0 = 0.9
        psi0 = np.sqrt(1.0 - r0 * r0)
        dpsi0 = -r0 / psi0
        psi_ufl = conditional(le(r, r0), sqrt(
            1.0 - r * r), psi0 + dpsi0 * (r - r0))
        psi = Function(V).interpolate(psi_ufl)
        # exact solution is known (and it determines Dirichlet boundary)
        afree = 0.697965148223374
        A = 0.680259411891719
        B = 0.471519893402112
        gbdry_ufl = conditional(le(r, afree), psi_ufl, -A * ln(r) + B)
        gbdry = Function(V).interpolate(gbdry_ufl)

        return psi, psi_ufl, gbdry

    def spiral_problem(mesh, V):
        # spiral obstacle problem from section 7.1.1 in Graeser & Kornhuber (2009)
        (x, y) = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        theta = atan2(y, x)
        tmp = sin(2.0*pi/r + pi/2.0 - theta) + r * \
            (r+1) / (r - 2.0) - 3.0 * r + 3.6
        psi = Function(V).interpolate(conditional(le(r, 1.0e-8), 3.6, tmp))
        gbdry = Constant(0.0)

        return psi, tmp, gbdry

    def getObstacle(self, V):
        mesh = V.mesh()
        if self.problem == "sphere":
            lb, psi_ufl, exact = self.sphere_problem(mesh, V)
        else:
            lb, psi_ufl, exact = self.spiral_problem(mesh, V)

        bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
        bcs = DirichletBC(V, exact, bdry_ids)

        return lb, bcs, exact

    def solveProblem(self, mesh=None, u=None):
        if mesh is None:
            mesh = self.getInitialMesh()
        # Fixme: possibility of higher order
        V = FunctionSpace(mesh, "CG", 1)

        if u is None:
            u = Function(V, name="u (FE soln)")
        else:
            # use cross mesh interp to ensure solution on current mesh
            u = Function(V, name="u (FE soln)").interpolate(u)

        lb, bcs, _ = getObstacle(V)

        # weak form problem; F is residual operator in nonlinear system F==0
        v = TestFunction(V)
        # as in Laplace equation:  - div (grad u) = 0
        F = inner(grad(u), grad(v)) * dx

        sp = {"snes_monitor": None,
              "snes_type": "vinewtonrsls",
              "snes_converged_reason": None,
              "snes_rtol": 1.0e-8,
              "snes_atol": 1.0e-12,
              "snes_stol": 1.0e-12,
              "snes_vi_zero_tolerance": 1.0e-12,
              "snes_linesearch_type": "basic",
              "ksp_type": "preonly",
              "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps"}
        problem = NonlinearVariationalProblem(F, u, bcs)
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="")

        # No upper obstacle
        ub = Function(V).interpolate(Constant(PETSc.INFINITY))
        solver.solve(bounds=(lb, ub))

        return u, lb
