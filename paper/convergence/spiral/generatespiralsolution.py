from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from paper.convergence.utility import SpiralObstacleProblem
import os
from fascd import FASCDSolver


width, offset = 2.0, -1.0  


base = SquareMesh(4, 4, width, distribution_parameters={
                                "partition": True,
                                "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
                            })
mh = MeshHierarchy(base, 8-1)
mesh = mh[-1]


for j in range(8):
    mh[j].coordinates.dat.data[:, :] += offset

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
ub = None
gbdry = Constant(0.0)


# operator 
u = Function(V, name="u (FE soln)")  # gets initialized to max(0.0,psi) inside FASCDSolver
J = 0.5*inner(grad(u), grad(u)) * dx  # implies  F=inner(grad(u),grad(v))*dx
v = TestFunction(V)
F = derivative(J, u, v)


bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
bcs = DirichletBC(V, gbdry, bdry_ids)

problem = NonlinearVariationalProblem(F, u, bcs)


sp = {"fascd_monitor": None,
          "fascd_converged_reason": None,
          "fascd_levels_snes_type": "vinewtonrsls",
          "fascd_levels_snes_max_it": 1,
          "fascd_levels_snes_vi_zero_tolerance": 1.0e-12,
          "fascd_levels_snes_linesearch_type": "basic",
          "fascd_levels_ksp_type": "cg",
          "fascd_levels_ksp_max_it": 3,
          "fascd_levels_ksp_converged_maxits": None,
          "fascd_levels_pc_type": "bjacobi",
          "fascd_levels_sub_pc_type": "icc",
          "fascd_coarse_snes_rtol": 1.0e-8,
          "fascd_coarse_snes_atol": 1.0e-12,
          "fascd_coarse_snes_stol": 1.0e-12,
          "fascd_coarse_snes_type": "vinewtonrsls",
          "fascd_coarse_snes_vi_zero_tolerance": 1.0e-12,
          "fascd_coarse_snes_linesearch_type": "basic",
          "fascd_coarse_ksp_type": "preonly",
          "fascd_coarse_pc_type": "lu",
          "fascd_coarse_pc_factor_mat_solver_type": "mumps"}
solver = FASCDSolver(problem, solver_parameters=sp, options_prefix="", bounds=(lb, ub),
                         warnings_convergence=True)
solver.solve()


ExactMesh = mh[-1]
ExactU = u
ExactU.rename("ExactU")
VTKFile("test.pvd").write(ExactU)
# Open issue won't run in parallel with netgen mesh: https://github.com/firedrakeproject/firedrake/issues/3783
with CheckpointFile("ExactSolutionSpiral.h5", "w") as afile:
    afile.save_mesh(ExactMesh)
    afile.save_function(ExactU)








# For debugging purposes
# os.chdir("/home/stefano/Desktop/VI-AMR/NumericalResults/ConvergenceResults")
# sp_fascd = {
#             "fascd_cycle_type": "full",
#             "fascd_skip_downsmooth": None,
#             "fascd_monitor": None,
#             "fascd_converged_reason": None,
#             "fascd_levels_snes_type": "vinewtonrsls",
#             "fascd_levels_snes_max_it": 3,
#             "fascd_levels_snes_converged_reason": None,
#             "fascd_levels_snes_vi_zero_tolerance": 1.0e-12,
#             "fascd_levels_snes_linesearch_type": "basic",
#             "fascd_levels_ksp_type": "cg",
#             "fascd_levels_ksp_max_it": 3,
#             "fascd_levels_ksp_converged_maxits": None,
#             "fascd_levels_pc_type": "bjacobi",
#             "fascd_levels_sub_pc_type": "icc",
#             "fascd_coarse_snes_converged_reason": None,
#             "fascd_coarse_snes_rtol": 1.0e-8,
#             "fascd_coarse_snes_atol": 1.0e-12,
#             "fascd_coarse_snes_stol": 1.0e-12,
#             "fascd_coarse_snes_type": "vinewtonrsls",
#             "fascd_coarse_snes_vi_zero_tolerance": 1.0e-12,
#             "fascd_coarse_snes_linesearch_type": "basic",
#             "fascd_coarse_ksp_type": "preonly",
#             "fascd_coarse_pc_type": "lu",
#             "fascd_coarse_pc_factor_mat_solver_type": "mumps"
#             
#         }
#         
# problem = SpiralObstacleProblem(spFASCD = sp_fascd, TriHeight=.1)
# amr = VIAMR()
# ExactMesh = problem.setInitialMesh()
# ExactU = None
# 
# nv, ne, hmin, hmax = amr.meshsizes(ExactMesh)
# for i in range(4):
#     ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU)
#     ExactU.rename("ExactU")
#     if i == 0:
#         # Run one metric refine to have substantially different mesh from convergence runs
#         amr.setmetricparameters(target_complexity=nv)
#         ExactMesh = amr.adaptaveragedmetric(
#             ExactMesh, ExactU, lb
#         )
#     else:
#         DG0 = FunctionSpace(ExactMesh, "DG", 0)
#         resUFL = -div(grad(ExactU))
#         FBmark = amr.vcdmark(ExactU, lb)
#         (BRmark, _, _) = amr.brinactivemark(ExactU, lb, resUFL, theta=0.5)
#         mark = amr.unionmarks(FBmark, BRmark)
#         ExactMesh = amr.refinemarkedelements(ExactMesh, mark)
# 
# 
# mh = MeshHierarchy(ExactMesh, 2)
# ExactMesh = mh[-1]
# 
# ExactU = Function(FunctionSpace(ExactMesh, "CG", 1)).interpolate(ExactU)
# 
# ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU, FASCD=True)
# ExactU.rename("ExactU")
# 
# # Open issue won't run in parallel with netgen mesh: https://github.com/firedrakeproject/firedrake/issues/3783
# with CheckpointFile("ExactSolutionSpiral.h5", "w") as afile:
#     afile.save_mesh(ExactMesh)
#     afile.save_function(ExactU)
# 