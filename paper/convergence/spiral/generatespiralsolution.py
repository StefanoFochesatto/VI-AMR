from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from paper.convergence.utility import SpiralObstacleProblem
import os

# For debugging purposes
# os.chdir("/home/stefano/Desktop/VI-AMR/NumericalResults/ConvergenceResults")
sp_fascd = {
            "fascd_cycle_type": "full",
            "fascd_skip_downsmooth": None,
            "fascd_monitor": None,
            "fascd_converged_reason": None,
            "fascd_levels_snes_type": "vinewtonrsls",
            "fascd_levels_snes_max_it": 3,
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
            "fascd_coarse_pc_factor_mat_solver_type": "mumps"
            
        }
        
problem = SpiralObstacleProblem(spFASCD = sp_fascd, TriHeight=.1)
amr = VIAMR()
ExactMesh = problem.setInitialMesh()
ExactU = None

nv, ne, hmin, hmax = amr.meshsizes(ExactMesh)
for i in range(4):
    ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU)
    ExactU.rename("ExactU")
    if i == 0:
        # Run one metric refine to have substantially different mesh from convergence runs
        amr.setmetricparameters(target_complexity=nv)
        ExactMesh = amr.adaptaveragedmetric(
            ExactMesh, ExactU, lb
        )
    else:
        DG0 = FunctionSpace(ExactMesh, "DG", 0)
        resUFL = -div(grad(ExactU))
        FBmark = amr.vcdmark(ExactU, lb)
        (BRmark, _, _) = amr.brinactivemark(ExactU, lb, resUFL, theta=0.5)
        mark = amr.unionmarks(FBmark, BRmark)
        ExactMesh = amr.refinemarkedelements(ExactMesh, mark)


mh = MeshHierarchy(ExactMesh, 2)
ExactMesh = mh[-1]

ExactU = Function(FunctionSpace(ExactMesh, "CG", 1)).interpolate(ExactU)

ExactU, lb = problem.solveProblem(mesh=ExactMesh, u=ExactU, FASCD=True)
ExactU.rename("ExactU")

# Open issue won't run in parallel with netgen mesh: https://github.com/firedrakeproject/firedrake/issues/3783
with CheckpointFile("ExactSolutionSpiral.h5", "w") as afile:
    afile.save_mesh(ExactMesh)
    afile.save_function(ExactU)
