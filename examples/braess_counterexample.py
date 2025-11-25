from viamr import VIAMR
from firedrake import *


class BraessVIAMR(VIAMR):
    """Extended VIAMR class for Braess' counter example"""

    def __init__(self, **kwargs):
        # Call parent constructor
        super().__init__(**kwargs)


    def braess_edge_indicator(self, uh, lb, theta=0.5, method="total"):
        """
            Implements a modified version of the edge-based residual error indicator described in 
            Braess, Carstensen, & Hoppe (2007)
            
            The estimator relies on the jump residual contributions only. 
            eta_E = h_E^{1/2} * || jump(grad(uh)) ||_{0, E}.
            """
        # mesh quantities
        mesh = uh.function_space().mesh()
        h = CellDiameter(mesh)
        v = CellVolume(mesh)
        n = FacetNormal(mesh)

        # Our refinement code requires element-wise indicators.
        # So the modification splits edge residuals in half to be assigned to the two adjacent cells.
        _, DG0 = self.spaces(mesh)
        eta_sq = Function(DG0)
        w = TestFunction(DG0)

        # BRAESS MODIFICATION from BR Indicator:
        # We REMOVE the cell residual term: - inner(h**2 * res_ufl**2, w) * dx
        # We only keep the edge jumps (interior facets dS).
        G = (
            inner(eta_sq / v, w) * dx
            # Project half the edge error to the (+) cell
            - inner(h("+") / 2 * jump(grad(uh), n) ** 2, w("+")) * dS
            # Project half the edge error to the (-) cell
            - inner(h("-") / 2 * jump(grad(uh), n) ** 2, w("-")) * dS
        )


        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)
        
        # This is computed over the entire domain.... 
        eta = Function(DG0).interpolate(sqrt(eta_sq))
        
        # The bulk criterion marking is essentially a fixed rate of the total strategy. 
        # Greedy as in marking largest errors first and that is what is implemented in _fixedrate with method = "total".
        mark, _, total_error_est = self._fixedrate(eta, theta, method)

        return (mark, eta, total_error_est)
