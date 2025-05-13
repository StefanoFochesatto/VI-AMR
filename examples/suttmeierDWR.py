# Solve the VI problem in section 10.3 of
#   F.-T. Suttmeier (2008).  Numerical Solution of Variational Inequalities
#   by Adaptive Finite Elements, Vieweg + Teubner, Wiesbaden
# Instead of using their estimator, we use VCD + DWR estimator over the inactive set. 

from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
import numpy as np
from viamr import VIAMR

print = PETSc.Sys.Print  # enables correct printing in parallel

refinements = 6
m0 = 10
outfile = "result_suttmeier.pvd"

params = {
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-12,
    "snes_linesearch_type": "basic",
    # "snes_monitor": None,
    # "snes_vi_monitor": None,
    "snes_converged_reason": None,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-12,
    "snes_stol": 1.0e-12,
    "snes_max_it": 200,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

meshhierarchy = [
    UnitSquareMesh(m0, m0),
]
amr = VIAMR()
for i in range(refinements + 1):
    mesh = meshhierarchy[i]
    print(f"solving on mesh {i} ...")
    amr.meshreport(mesh)

    # initial iterate by cross-mesh interpolation from coarser mesh
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u_h").interpolate(Constant(0.0) if i == 0 else u)

    # problem data
    x, y = SpatialCoordinate(mesh)
    psi = Function(V, name="psi").interpolate(
        -(((x - 0.5) ** 2 + (y - 0.5) ** 2) ** (3 / 2))
    )
    # typo? from Suttmeier: f = 10.0 * (x - x**2 + y - y **2)
    fsource = Function(V, name="f").interpolate(-10.0 * (x - x ** 2 + y - y ** 2))

    # weak form and problem
    v = TestFunction(V)
    F = inner(grad(u), grad(v)) * dx - fsource * v * dx
    bcs = DirichletBC(V, Constant(0.0), "on_boundary")
    problem = NonlinearVariationalProblem(F, u, bcs)

    # solve the VI
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=params, options_prefix="s"
    )
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(psi, ub))

    print(f"u_h(1/8,1/4) = {u.at(0.125, 0.25):.6e}")
    if i == refinements:
        break
    
    ## Adaptive Meshing Routine
    # Mark FB with vcd
    markFB = amr.vcdmark(mesh, u, psi, bracket=[0.1, 0.8])
    
    
    # Implementation of point value DWR Estimator over inactive set. 
    # This estimator is derived in Example 3.3 of Bangerth and Rannacher's Adaptive Finite Element Methods for Differential Equations.
    # Section 3.3 of this book discusses how the DWR estimator is simplified for certain choices of output functionals namely the point value 
    # derivative value functionals. With these problems there is no need to compute compute a numerical dual  (at least thats what they suggest)
    # the main way this is done is through the approximation just above equation (3.14) 
    # \\[\omega_K \approx h_K^2||\nabla^2 z||_K \approx h_K^2 |K|^{1/2}r_K^{-2} \quad r_K:=max_{x \in K} \sqrt{|x - a| + \epsilon^2} \quad \epsilon := TOL\\]
    
    # The justification seems to be  that z behaves like a regularized Green function
    #\\[z(x) \approx g_\epsilon^a(x) \approx \log(|x - a| + \epsilon^2)\\]
    
    # Submesh inactive set
    inactiveIndicator = amr.eleminactive(u, psi)
    mesh.mark_entities(inactiveIndicator, 999)
    testmesh = RelabeledMesh(mesh, [inactiveIndicator], [999])
    inactivemesh= Submesh(testmesh, 2, 999)

    # The element wise computation on K is:
    # \\[(h_K^3/r_K^2)\rho_K \\]
    # Recall that \rho_K is essentially the BR estimator
    DG0 = FunctionSpace(inactivemesh, "DG", 0)
    CG1 = FunctionSpace(inactivemesh, "CG", 1)
    
    epsilon = 1.0e-5 # TOL
    a = as_vector([0.2, 0.8])  # Point
    x = SpatialCoordinate(inactivemesh)
    r = sqrt(norm(x-a) + epsilon**2)
    
    rk = Function(CG1).interpolate(r)
    HK = Function(DG0).interpolate(CellDiameter(inactivemesh))
    
    RK = Function(DG0)
    
    #for i in range(RK.dat.data.shape[0]):
        
    
    
    
    
    
    
        

    _, DG0 = amr.spaces(mesh)
    mark = Function(DG0).interpolate((mark + imark) - (mark * imark))  # union

    mesh = amr.refinemarkedelements(mesh, mark)
    meshhierarchy.append(mesh)

print(f"done ... writing u_h, psi, f, gap=u_h-psi to {outfile} ...")
gap = Function(V, name="gap = u_h - psi").interpolate(u - psi)
VTKFile(outfile).write(u, psi, fsource, gap)
