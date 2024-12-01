import numpy as np
from firedrake import *
from firedrake.petsc import OptionsManager
from firedrake.output import VTKFile


class VIAMR(OptionsManager):
    def __init__(self, **kwargs):
        # solver_parameters = flatten_parameters(kwargs.pop("solver_parameters",{}))
        # self.options_prefix = kwargs.pop("options_prefix","")
        # super().__init__(solver_parameters, self.options_prefix)  # set PETSc parameters
        pass

    def vcesmark(self, cmesh, u, lb, tolerance=1e-10, bracket=[0.2, 0.8]):
        "Mark current mesh using Variable Coefficient Elliptic Smoothing (VCES) algorithm"

        # Compute nodal active set indicator within some tolerance
        CG1 = FunctionSpace(cmesh, "CG", 1)
        nodalactive = Function(CG1, name="Nodal Active Set Indicator")
        nodalactive.interpolate(conditional(abs(u - lb) < tolerance, 0, 1))

        # Vary timestep by average cell area of each patch.
        # Not exactly an average because msh.cell_sizes is an L2 projection of
        # the obvious DG0 function into CG1.
        timestep = Function(CG1)
        # FIXME following not parallel
        timestep.dat.data[:] = 0.5 * cmesh.cell_sizes.dat.data[:] ** 2

        # Solve one step implicitly using a linear solver
        # Nodal indicator is initial condition to time dependent Heat eq
        w = TrialFunction(CG1)
        v = TestFunction(CG1)
        a = w * v * dx + timestep * inner(grad(w), grad(v)) * dx
        L = nodalactive * v * dx
        u = Function(CG1, name="Smoothed Nodal Active Indicator")
        solve(a == L, u)

        # Compute average over elements by interpolation into DG0
        DG0 = FunctionSpace(cmesh, "DG", 0)
        uDG0 = Function(DG0, name="Smoothed Nodal Active Indicator as DG0")
        uDG0.interpolate(u)

        # Applying thresholding parameters
        mark = Function(DG0, name="VCES Marking")
        mark.interpolate(
            conditional(uDG0 > bracket[0], conditional(uDG0 < bracket[1], 1, 0), 0)
        )
        return mark

    def udomark(self, cmesh):
        "Mark current mesh using Unstructured Dilation Operator (UDO) algorithm"
        raise NotImplementedError  # TODO
