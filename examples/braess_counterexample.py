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

        # Our refinement code requires element-wise indicators. It is more efficient than what is described after eq (11) in the paper and still generates conforming meshes. 
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
        
        
        # Comments:
        # The top of page 459 states. Moreover, the refinement and the new mesh T_{l + 1} shall also take care of of a reduction of the data oscillation

        return (mark, eta, total_error_est)
    
    
    def braess_oscillation(self, uh, lb, theta, method="total"):
        """
        Computes the Braess et al. estimator where data oscillation is 
        computed patch-wise over edges using algebraic expansion.
        """
        mesh = uh.function_space().mesh()
        DG0 = FunctionSpace(mesh, "DG", 0)
        w = TestFunction(DG0)
        n = FacetNormal(mesh)
        h = FacetArea(mesh)

        # ==========================================
        # 1. Pre-compute Cell Integrals for Oscillation
        # ==========================================
        # We need int_K(1), int_K(f), and int_K(f^2) for every cell K

        vol_k = Function(DG0, name="Cell Volumes")
        int_f_k = Function(DG0, name="Int f")
        int_f2_k = Function(DG0, name="Int f^2")

        # Project/Assemble these cell-wise quantities
        assemble(Constant(1.0) * w * dx, tensor=vol_k)
        assemble(f * w * dx, tensor=int_f_k)
        assemble(f**2 * w * dx, tensor=int_f2_k)

        # ==========================================
        # 2. Compute Oscillation on Edges (Eq 9)
        # ==========================================
        # Patch quantities using (+) and (-) neighbors
        # |Omega_E| = vol(+) + vol(-)
        patch_vol = vol_k('+') + vol_k('-')

        # int_{Omega_E} f = int_{K+} f + int_{K-} f
        patch_int_f = int_f_k('+') + int_f_k('-')

        # int_{Omega_E} f^2 = int_{K+} f^2 + int_{K-} f^2
        patch_int_f2 = int_f2_k('+') + int_f2_k('-')

        # Algebraic expansion of: |Omega_E| * || f - f_mean ||^2
        # osc_E^2 = |Omega_E| * int(f^2) - (int(f))^2
        osc_edge_sq = patch_vol * patch_int_f2 - patch_int_f**2

        # Distribute 50/50 to neighbors for element-wise visualization/marking
        osc_dist_form = 0.5 * osc_edge_sq * (w('+') + w('-')) * dS

        # Add boundary term for Equation 10 (h_T^2 * ||f||_T^2 on boundary elements)
        # The paper implies T_F are elements touching the boundary.
        # We use 'ds' to identify boundary elements.
        h_cell = CellDiameter(mesh)
        # Note: Eq 10 uses h_T^2 * ||f||^2.
        # ||f||^2_T is exactly int_f2_k we computed earlier (as a DG0 coeff)
        # We only apply this where ds exists.
        # Strategy: Assemble into a boundary-only vector or add to form
        osc_boundary_form = (h_cell**2 * f**2 * w) * ds

        # Final Oscillation Vector
        osc_sq = Function(DG0, name="Squared Data Oscillation")
        assemble(osc_dist_form + osc_boundary_form, tensor=osc_sq)
        osc = Function(DG0).interpolate(sqrt(osc_sq))
