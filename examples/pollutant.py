import time
import sys
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
print = PETSc.Sys.Print # enables correct printing in parallel
from firedrake import *
import petsc4py
from viamr import VIAMR
from netgen.occ import *

# 3D advection-diffusion problem with upper and lower bounds

# This issue about refinement along the surface will be fixed in the next firedrake release. Install the latest ngsPETSc changes for now. 

refinements = 3   # serial run will be many minutes for this value

# initial mesh
box = Box((-1, -1, -1), (1, 1, 1))
ngmesh = OCCGeometry(box, dim=3).GenerateMesh(maxh=.2)
mesh = Mesh(ngmesh, distribution_parameters={
            "partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)})

amr = VIAMR()
for i in range(refinements + 1):
    amr.meshreport(mesh)
    # obstacle and solution are in P1 or Q1
    V = FunctionSpace(mesh, "CG", 1)
    Z = VectorFunctionSpace(mesh, "CG", 1)  # velocity in this space

    # parameters
    alpha = 30.0  # strength of positive source
    beta = 4.0    # strength of negative removal

    # fields: w = velocity, f = source
    x0, y0, z0, r0 = -0.5,  0.5, 0.5, 1.0/3.0
    x1, y1, z1, r1 = -0.5, -0.5, -0.2, 1.0/5.0

    # generate source term from formulas found in
    #   https://bitbucket.org/pefarrell/fascd/src/master/examples/pollutant.py
    # see: https://doi.org/10.1137/23M1594200
    (x, y, z) = SpatialCoordinate(mesh)
    w = Function(Z, name="w (velocity)").interpolate(as_vector([7.0 + 5.0 * y, -5.0 * x, 2.0 * z]))
    disc0 = (x-x0) * (x-x0) + (y-y0) * (y-y0) + (z-z0) * (z-z0)
    disc1 = (x-x1) * (x-x1) + (y-y1) * (y-y1) + (z-z1) * (z-z1)
    left_ufl = conditional(Or(disc0 < r0**2, disc1 < r1**2), alpha, 0.0)
    right_ufl = conditional(x > 0, -beta * (1.0 - cos(6.0*pi*x)), 0.0)
    f = Function(V, name="f (source)").interpolate(left_ufl + right_ufl)

    u = Function(V, name="u (FE soln)").interpolate(Constant(.5))  # initialized to 0.0, so admissible
    v = TestFunction(V)
    F = Constant(0.1) * inner(grad(u), grad(v)) * dx + \
        inner(dot(w, grad(u)) - f, v) * dx
    bdry_ids = (1,)   # Dirichlet only on x = -1 side
    bcs = DirichletBC(V, Constant(0.0), bdry_ids)
    problem = NonlinearVariationalProblem(F, u, bcs)

    # Solve
    sp = {
        "snes_type": "vinewtonrsls",
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-12,
        "snes_stol": 1.0e-12,
        "snes_vi_zero_tolerance": 1.0e-12,
        "snes_converged_reason": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    lb = Function(V).interpolate(Constant(0.0))
    ub = Function(V).interpolate(Constant(1.0))

    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="")
    solver.solve(bounds=(lb, ub))

    if i == refinements:
        break

    marklower = amr.udomarkParallel(mesh, u, lb, n = 1)
    markupper = amr.udomarkParallel(mesh, u, ub, n = 1)
    _, DG0 = amr.spaces(mesh)
    mark = Function(DG0).interpolate((marklower + markupper) - (marklower * markupper))
    mesh = mesh.refine_marked_elements(mark)
    #VTKFile(f'test{i}.pvd').write(u, f, w, lb, ub, marklower, markupper, mark)

outfile = 'result_pollutant.pvd'
print(f'done ... writing to {outfile} ...')
if mesh.comm.size > 1:  # integer-valued element-wise process rank
    rank = Function(FunctionSpace(mesh,'DG',0))
    rank.dat.data[:] = mesh.comm.rank
    rank.rename('rank')
    VTKFile(outfile).write(u, f, w, rank)
else:
    VTKFile(outfile).write(u, f, w)
