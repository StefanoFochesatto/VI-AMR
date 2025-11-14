from firedrake import *
from firedrake.petsc import PETSc
from viamr import VIAMR
import numpy as np

from animate import *
print = PETSc.Sys.Print


class AdaptiveVISolver:
    def __init__(self, m0=32, T=1.0, dt=0.01, target_complexity=100.0, h_min=1.0e-6, h_max=1):
        """
        Initialize the adaptive variational inequality solver.
        
        Args:
            m0: Initial mesh resolution
            T: Final time
            dt: Time step size
            target_complexity: Target complexity for mesh adaptation
            h_min: Minimum mesh size
            h_max: Maximum mesh size
        """
        # Time parameters
        self.dt = dt
        self.T = T
        self.num_steps = int(T / dt)
        self.t = 0.0
        
        # Problem parameters
        self.t_param = Constant(0.0)
        self.sigma = Constant(0.09)
        self.tfactor = Constant(0.00005)
        self.height = Constant(0.005)
        
        # Mesh and adaptation
        self.mesh = RectangleMesh(m0, m0, Lx=1.0, Ly=1.0, originX=-1.0, originY=-1.0)
        self.amr = VIAMR()
        
        # Adaptation parameters
        self.target_complexity = target_complexity
        self.h_min = h_min
        self.h_max = h_max
        self.metric_buffer = []
        
        # Set metric refinement parameters
        self.amr.setmetricparameters()
        
        # Solver parameters for VI
        self.solver_params = {
            "snes_type": "vinewtonrsls",
            "snes_vi_zero_tolerance": 1.0e-12,
            "snes_linesearch_type": "basic",
            "snes_max_it": 200,
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-12,
            "snes_stol": 0.0,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_converged_reason": None,
        }
        
        # Output file
        self.outfile = VTKFile("result_sphere_timedep_buffered.pvd", adaptive=True)
        
        # Solution variables
        self.u = None
        self.u_n = None
        self.V = None
        self.lb = None
    def setup_problem(self, prev_mesh_u_n=None):
        """Setup function space and boundary conditions on current mesh."""
        self.V = FunctionSpace(self.mesh, "CG", 1)
        
        # Initialize or interpolate previous solution
        if self.u_n is None:
            # Initial condition (zero)
            self.u_n = Function(self.V, name="u_n")
        elif prev_mesh_u_n is not None:
            # Cross-mesh interpolation from previous mesh
            self.u_n = Function(self.V).interpolate(prev_mesh_u_n)
        
        # Current solution
        self.u = Function(self.V, name="u")
        
        # Homogeneous Dirichlet boundary conditions
        self.bcs = DirichletBC(self.V, 0.0, "on_boundary")
        
    def update_source_term(self):
        """Update the time-dependent source term."""
        x, y = SpatialCoordinate(self.mesh)
        f_ufl = self.height * (self.t_param * self.tfactor) * exp(-((x)**2 + (y)**2) / (self.sigma**2))
        self.f = Function(self.V, name="f_source").interpolate(f_ufl)
        
    def solve_step(self, replace_u_n=False, update_t=False, write=False):
        """
        Solve one time step of the VI problem.
        
        Args:
            replace_u_n: Whether to update u_n with current solution
            update_t: Whether to increment time
            write: Whether to write output
        """
        # Update source term for current time
        self.update_source_term()
        
        # Weak form using backward Euler
        v = TestFunction(self.V)
        F = ((self.u - self.u_n) / self.dt * v + inner(grad(self.u), grad(v)) - self.f * v) * dx
        
        # Set up the VI problem
        problem = NonlinearVariationalProblem(F, self.u, self.bcs)
        solver = NonlinearVariationalSolver(problem, solver_parameters=self.solver_params)
        
        # Obstacle constraint: u >= 0
        self.lb = Function(self.V, name="obstacle").assign(0.0)
        ub = Function(self.V).assign(Constant(PETSc.INFINITY))
        
        # Solve with bounds
        solver.solve(bounds=(self.lb, ub))
        
        # Write output if requested
        if write:
            active_set = self.amr.eleminactive(self.u, self.lb)
            self.outfile.write(self.u, active_set, time=self.t)
        
        # Update time if requested
        if update_t:
            self.t += self.dt
            self.t_param.assign(self.t)
        
        # Update u_n if requested
        if replace_u_n:
            self.u_n.interpolate(self.u)
    
    def get_metric(self):
        """Compute metric from current solution and obstacle."""
        lb = Function(self.V).assign(0.0)
        metric = self.amr.adaptaveragedmetric(self.mesh, self.u, self.lb, return_metric=True)
        return metric
    
    def run_with_buffer(self, buffer_size=10):
        """
        Run the time-stepping loop with buffered adaptation.
        
        Args:
            buffer_size: Number of steps between mesh adaptations
        """
        self.setup_problem()
        
        for step in range(self.num_steps + 1):
            # Initial adaptation step
            if step == 0:
                self.t_param.assign(0.0)
                self.solve_step(replace_u_n=False, update_t=False, write=True)
                # Initial mesh adaptation
                self.lb = Function(self.V).assign(0.0)
                self.mesh = self.amr.adaptaveragedmetric(self.mesh, self.u, self.lb)
                self.setup_problem(prev_mesh_u_n=self.u_n)
                continue
            
            # Every buffer_size steps, perform mesh adaptation
            if step % buffer_size == 0:
                # Store current solution for interpolation
                store_u = self.u.copy(deepcopy=True)
                
                # Fill metric buffer
                self.metric_buffer = []
                for _ in range(buffer_size):
                    self.solve_step(replace_u_n=True, update_t=False, write=False)
                    metric = self.amr.adaptaveragedmetric(self.mesh, self.u_n, self.lb, metric=True, gamma = 1, bracket=[0.1,0.9] )
                    self.metric_buffer.append(metric)

                metricAVG = self.metric_buffer[0].copy(deepcopy=True)
                metricAVG.average(*self.metric_buffer[1:])
                metricAVG.normalise()
                
                
                
                # Adapt mesh based on last solution in buffer
                self.lb = Function(self.V).assign(0.0)
                self.mesh = adapt(self.mesh, metricAVG)
                
                # Setup problem on new mesh, interpolating stored solution
                self.setup_problem(prev_mesh_u_n=store_u)
            
            # Solve on current (possibly adapted) mesh
            self.solve_step(replace_u_n=True, update_t=True, write=True)
            
            if step % 10 == 0:
                print(f"Step {step}/{self.num_steps}, t = {self.t:.4f}")
    
    def run_simple(self):
        """Run with adaptation at every time step (original approach)."""
        self.setup_problem()
        
        for step in range(self.num_steps + 1):
            self.t = step * self.dt
            self.t_param.assign(self.t)
            
            # Solve current step
            self.solve_step(replace_u_n=False, update_t=False, write=True)
            
            # Adapt mesh
            self.lb = Function(self.V).assign(0.0)
            self.mesh = self.amr.adaptaveragedmetric(self.mesh, self.u, self.lb)
            
            # Setup for next step
            prev_u_n = self.u_n.copy(deepcopy=True) if self.u_n is not None else None
            self.setup_problem(prev_mesh_u_n=self.u)
            
            if step % 10 == 0:
                print(f"Step {step}/{self.num_steps}, t = {self.t:.4f}")


# Main execution
if __name__ == "__main__":
    # Run with buffered adaptation (every 10 steps)
    solver = AdaptiveVISolver(m0=25, T=1.0, dt=0.01)
    solver.run_with_buffer(buffer_size=5)
    
    # Or run with adaptation at every step (original approach)
    # solver = AdaptiveVISolver(m0=32, T=1.0, dt=0.01)
    # solver.run_simple()