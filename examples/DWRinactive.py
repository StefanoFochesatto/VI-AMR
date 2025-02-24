from firedrake import *
from firedrake.output import VTKFile
from viamr import VIAMR
from viamr.utility import LShapedDomainProblem
levels = 4
outfile = "result_LShaped.pvd"



# Solve Obstacle Problem over domain,
# Mesh filtering functionality in dmplextransform to setup inactive dirichlet problem? Looks like matt is working on porting to firedrake
# Compute error estimate
# Union FB and DWR marks


def estimate_error(mesh, uh):
    W = FunctionSpace(mesh, "DG", 0)
    eta_sq = Function(W)
    w = TestFunction(W)
    f = Constant(0.0)
    h = CellDiameter(mesh)  # symbols for mesh quantities
    n = FacetNormal(mesh)
    v = CellVolume(mesh)

    # Babuska-Rheinboldt error estimator
    G = (  # compute cellwise error estimator
        inner(eta_sq / v, w)*dx
        - inner(h**2 * (f + div(grad(uh)))**2, w) * dx
        - inner(h('+')/2 * jump(grad(uh), n)**2, w('+')) * dS
        - inner(h('-')/2 * jump(grad(uh), n)**2, w('-')) * dS
    )

    # Each cell is an independent 1x1 solve, so Jacobi is exact
    sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
    solve(G == 0, eta_sq, solver_parameters=sp)
    eta = Function(W).interpolate(sqrt(eta_sq))  # compute eta from eta^2

    with eta.dat.vec_ro as eta_:  # compute estimate for error in energy norm
        error_est = sqrt(eta_.dot(eta_))
    return (eta, error_est)








if __name__ == "__main__":
    problem = LShapedDomainProblem(TriHeight=.25)
    amr = VIAMR()
    mesh = problem.setInitialMesh()
    meshHist = [mesh]
    u = None
    for i in range(levels + 1):
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
        
        # Generate Inactive DWR mark
        (eta, error_est) = estimate_error(mesh, u)
        CG1, DG0 = amr.spaces(mesh)
        elemactiveindicator = amr.elemactive(u, lb)
        eleminactiveindicator = Function(DG0).interpolate(conditional(elemactiveindicator>0.5, 0, 1))
        etaInactive = Function(DG0).interpolate(eta * eleminactiveindicator)
        etaInactive.rename("etaInactive")
        markDWR = Function(DG0)
        with etaInactive.dat.vec_ro as eta_:
            eta_max = eta_.max()[1]

        theta = 0.5
        should_refine = conditional(gt(etaInactive, theta*eta_max), 1, 0)
        markDWR.interpolate(should_refine)
        
        # Generate Free Boundary mark
        markFB = amr.vcesmark(mesh, u, lb, bracket=[.4, .6])
        
        # Union and Refine
        markUnion = Function(DG0).interpolate((markDWR + markFB) - (markDWR*markFB))
        mesh = mesh.refine_marked_elements(markUnion)
        meshHist.append(mesh)

    V = u.function_space()
    gap = Function(V, name="gap = u-lb").interpolate(u - lb)


    VTKFile(outfile).write(u, lb, gap)
