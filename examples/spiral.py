# This example generates two .pvd files, result_spiral_{udo,vcd}.pvd, suitable for
# a figure in the paper comparing n=1 UDO to [0.1,0.9] VCD on the spiral problem.
# Note: Because of thin active set, jaccard agreement is zero until levels=4.

from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel
from viamr import VIAMR
from paper.convergence.utility import SpiralObstacleProblem

levels = 3
h_initial = 0.20

problem = SpiralObstacleProblem(TriHeight=h_initial)
amr = VIAMR()
spmore = {
    "snes_converged_reason": None,
    #"snes_vi_monitor": None,
}

for amrtype in ("udo", "vcd"):
    mesh = problem.setInitialMesh()
    meshHist = [mesh]
    u = None

    for i in range(levels + 1):
        mesh = meshHist[i]
        print(f"solving spiral problem with {amrtype} AMR on mesh {i} ...")
        amr.meshreport(mesh)
        u, lb = problem.solveProblem(mesh=mesh, u=u, moreparams=spmore)

        neweactive = amr._elemactive(u, lb)
        if i > 0:
            jac = amr.jaccard(neweactive, eactive, submesh=True)
            print(f"  Jaccard agreement {100*jac:.2f}% [levels {i-1}, {i}]")
        eactive = neweactive

        if i == levels:
            break

        residual = -div(grad(u))
        (imark, _, _) = amr.brinactivemark(u, lb, residual)
        if amrtype == "udo":
            mark = amr.udomark(u, lb, n=1)
        elif amrtype == "vcd":
            mark = amr.vcdmark(u, lb, bracket=[0.1, 0.9])
        else:
            raise ValueError("unknown amrtype")
        mark = amr.unionmarks(mark, imark)

        mesh = mesh.refine_marked_elements(mark)
        meshHist.append(mesh)

    outfile = "result_spiral_" + amrtype + ".pvd"
    print(f"done ... writing to {outfile} ...")
    V = u.function_space()
    gap = Function(V, name="gap = u-lb").interpolate(u - lb)
    VTKFile(outfile).write(u, lb, gap)
    print("")
