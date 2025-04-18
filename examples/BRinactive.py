from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
print = PETSc.Sys.Print # enables correct printing in parallel
from viamr import VIAMR
from viamr.utility import LShapedDomainProblem

levels = 6
outfile = "result_LShaped.pvd"

# Solve Obstacle Problem over domain,
# Mesh filtering functionality in dmplextransform to setup inactive dirichlet problem? Looks like matt is working on porting to firedrake
# Compute error estimate by BR, see VIAMR.br_estimate_error_poisson()
# Union FB and DWR marks

problem = LShapedDomainProblem(TriHeight=.25)
amr = VIAMR()
mesh = problem.setInitialMesh()
meshHist = [mesh]
u = None
for i in range(levels + 1):
    # solve problem on current mesh
    mesh = meshHist[i]
    print(f'solving on mesh {i} ...')
    amr.meshreport(mesh)
    spmore = {
        "snes_converged_reason": None,
        "snes_vi_monitor": None,
    }
    u, lb = problem.solveProblem(mesh=mesh, u=u, moreparams=spmore)
    if i == levels:
        break

    # generate inactive BR marking
    (markBR, _, _) = amr.br_mark_poisson(u, lb)

    # generate free boundary marking by VCD
    markFB = amr.vcdmark(mesh, u, lb, bracket=[.4, .6])

    # union markings, and refine
    _, DG0 = amr.spaces(mesh)
    markUnion = Function(DG0).interpolate((markBR + markFB) - (markBR * markFB))
    mesh = mesh.refine_marked_elements(markUnion)
    meshHist.append(mesh)

V = u.function_space()
gap = Function(V, name="gap = u-lb").interpolate(u - lb)

print(f'done ... writing solution u, obstacle lb, and gap to {outfile} ...')
VTKFile(outfile).write(u, lb, gap)
