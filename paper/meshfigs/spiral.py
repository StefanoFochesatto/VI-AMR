from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from paper.convergence.utility import SpiralObstacleProblem
import os
from fascd import FASCDSolver
from viamr import VIAMR

width, offset = 2.0, -1.0  

amr = VIAMR()
u = None
TriHeight = 1
for i in range (10):
    if i == 0:
      mesh = RectangleMesh(
                            nx=int(4 / TriHeight),
                            ny=int(4 / TriHeight),
                            Lx=1,
                            Ly=1,
                            originX=-1,
                            originY=-1,
                            distribution_parameters={
                                "partition": True,
                                "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
                            },
                        )
    
    # obstacle and solution are in P1 or Q1
    V = FunctionSpace(mesh, "CG", 1)

    # obstacle psi(x,y)
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    frhs = None
    uexact = None
    Large = Function(V).interpolate(Constant(100.0))
    mLarge = Function(V).interpolate(Constant(-100.0))

    # spiral
    theta = atan2(y,x)
    tmp = sin(2.0*pi/r + pi/2.0 - theta) + r * (r+1) / (r - 2.0) - 3.0 * r + 3.6
    lb = Function(V).interpolate(conditional(le(r, 1.0e-8), 3.6, tmp))
    ub =  Function(V).interpolate(Constant(PETSc.INFINITY))
    gbdry = Constant(0.0)


    # operator 
    if u == None:
        u = Function(V, name="u (FE soln)")  # gets initialized to max(0.0,psi) inside FASCDSolver
    else:
        u = Function(V, name = "u (FE soln)").interpolate(u)
    
    
    J = 0.5*inner(grad(u), grad(u)) * dx  # implies  F=inner(grad(u),grad(v))*dx
    v = TestFunction(V)
    F = derivative(J, u, v)


    bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
    bcs = DirichletBC(V, gbdry, bdry_ids)

    problem = NonlinearVariationalProblem(F, u, bcs)


    #  sp_fascd = {"fascd_monitor": None,
    #            "fascd_converged_reason": None,
    #            "fascd_levels_snes_type": "vinewtonrsls",
    #            "fascd_levels_snes_max_it": 1,
    #            "fascd_levels_snes_vi_zero_tolerance": 1.0e-12,
    #            "fascd_levels_snes_linesearch_type": "basic",
    #            "fascd_levels_ksp_type": "cg",
    #            "fascd_levels_ksp_max_it": 3,
    #            "fascd_levels_ksp_converged_maxits": None,
    #            "fascd_levels_pc_type": "bjacobi",
    #            "fascd_levels_sub_pc_type": "icc",
    #            "fascd_coarse_snes_rtol": 1.0e-8,
    #            "fascd_coarse_snes_atol": 1.0e-12,
    #            "fascd_coarse_snes_stol": 1.0e-12,
    #            "fascd_coarse_snes_type": "vinewtonrsls",
    #            "fascd_coarse_snes_vi_zero_tolerance": 1.0e-12,
    #            "fascd_coarse_snes_linesearch_type": "basic",
    #            "fascd_coarse_ksp_type": "preonly",
    #            "fascd_coarse_pc_type": "lu",
    #            "fascd_coarse_pc_factor_mat_solver_type": "mumps"}
    #  solver = FASCDSolver(problem, solver_parameters=sp_fascd, options_prefix="", bounds=(lb, ub),
    #                           warnings_convergence=True)

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
        "pc_factor_mat_solver_type": "mumps",
    }



    solver = NonlinearVariationalSolver(
        problem, solver_parameters=sp_default, options_prefix=""
    )

    solver.solve(bounds = (lb, ub))
    
    DG0 = FunctionSpace(mesh, "DG", 0)
    resUFL = -div(grad(u))
    FBmark = amr.udomark(u, lb, n  = 1)
    (BRmark, _, _) = amr.brinactivemark(u, lb, resUFL,method = "max", theta=0.95)
    mark = amr.unionmarks(FBmark, BRmark)
    mesh = amr.refinemarkedelements(mesh, mark)

active = amr.elemactive(u, lb)
VTKFile("spiral.pvd").write(active, u)
