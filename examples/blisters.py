from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
print = PETSc.Sys.Print # enables correct printing in parallel
import numpy as np
from viamr import VIAMR

levels = 2
#h_initial = 0.10
h_initial = 0.03
outfile = "result_blisters.pvd"

def normal2d(mesh, x0, y0, sigma):
    # return UFL expression for one gaussian hump
    x, y = SpatialCoordinate(mesh)
    C = 1.0 / (2.0 * pi * sigma**2)
    dsqr = (x - x0)**2 + (y - y0)**2
    return C * exp(- dsqr / (2.0 * sigma**2))

params = {  
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-12,
    "snes_linesearch_type": "basic",
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-12,
    "snes_stol": 1.0e-12,
    "snes_max_it": 200,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"}

m = int(np.ceil(1.0 / h_initial))
initial_mesh = UnitSquareMesh(m, m)

amr = VIAMR()
meshhierarchy = [initial_mesh]
for i in range(levels + 1):
    mesh = meshhierarchy[i]
    print(f'solving on mesh {i} ...')
    amr.meshreport(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    if i == 0:
        u = Function(V, name="u")
    else:
        # cross-mesh interpolation from coarser mesh
        u = Function(V, name="u").interpolate(u)

    f_ufl = -3.0 + 0.4 * normal2d(mesh, 0.3, 0.8, 0.06) \
                 + 0.5 * normal2d(mesh, 0.8, 0.35, 0.05) \
                 + 0.4 * normal2d(mesh, 0.8, 0.25, 0.06)
    fsource = Function(V, name='fsource').interpolate(f_ufl)

    v = TestFunction(V)
    F = inner(grad(u), grad(v)) * dx - fsource * v * dx
    bcs = DirichletBC(V, Constant(0.0), (1, 2, 3, 4))
    problem = NonlinearVariationalProblem(F, u, bcs)

    solver = NonlinearVariationalSolver(problem, solver_parameters=params, options_prefix="s")
    lb = Function(V).interpolate(Constant(0.0))
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    #spmore = {
    #    "snes_converged_reason": None,
    #    "snes_vi_monitor": None,
    #}
    solver.solve(bounds=(lb, ub))
    if i == levels:
        break
    mark = amr.vcesmark(mesh, u, lb, bracket=[0.2, 0.9])
    mesh = mesh.refine_marked_elements(mark)
    meshhierarchy.append(mesh)

outfile = 'result_blisters.pvd'
print(f'done ... writing to {outfile} ...')
VTKFile(outfile).write(u, fsource)
