  
Recall the definition of $osc_\ell(f)$ eq (9) in the paper:
$$osc_\ell(f) = \left(\sum_{E \in \mathcal{E}_\ell(\Omega)} osc^2_E(f)\right)^{1/2}$$
Note that $osc_E(f) = |\Omega_E|^{1/2} || f - f_{\Omega_E} ||_{0, \Omega_E}$. Here $\Omega_E$ is the patch of an edge $E$, so the two adjacent cells. We define 
$$
f_{\Omega_E} = \frac{1}{|\Omega_E|} \int_{\Omega_E} f \, dx
$$

        
Also recall the definition of $Osc_\ell(f)$ eq (10) in the paper:
$$
Osc_\ell(f) = \left( osc^2_\ell(f) + \sum_{\Tau \in T_\Gamma} h^2_\Tau ||f||^2_{0, \Tau}\right)^{1/2}
$$

Ensuring inequality (12) intelligently, 
$$
Osc_{\ell+1}(f) \leq \kappa  Osc_{\ell}(f)
$$

requires implementing a localized estimator for $Osc_\ell(f)$ and applying a bulk criterion refinement idea. (They do not describe how this is done in the paper either)

Squaring both sides of the inequality, we then only need to convert eq (9) into an element-wise definition to get a localized estimator. 

By definition 
$$
osc^2_E(f) = |\Omega_E|\int_{\Omega_E}(f - f_{\Omega_E})^2
$$
Expand the square and substitute the definition of $f_{\Omega_E}$ we get, 
$$
\int_{\Omega_E}(f - f_{\Omega_E})^2 = \int_{\Omega_E} f^2 - 2\int_{\Omega_E} f_{\Omega_E}f +  \int_{\Omega_E}f^2_{\Omega_E}
$$

$$
= \int_{\Omega_E} f^2 - 2\left(\frac{\int_{\Omega_E}f}{|\Omega_E|}\right) \int_{\Omega_E} f + \left(\frac{\int_{\Omega_E}f}{|\Omega_E|}\right)^2 |\Omega_E|
$$


$$
= \int_{\Omega_E} f^2 - \frac{2}{|\Omega_E|} \left(\int_{\Omega_E} f\right)^2 + \frac{1}{|\Omega_E|} \left(\int_{\Omega_E} f\right)^2
$$

$$
= \int_{\Omega_E} f^2 - \frac{1}{|\Omega_E|} \left(\int_{\Omega_E} f\right)^2
$$

Now substitute this back into the definition for the edge oscillation $osc^2_E(f)$:

$$
osc^2_E(f) = |\Omega_E| \left( \int_{\Omega_E} f^2 - \frac{1}{|\Omega_E|} \left(\int_{\Omega_E} f\right)^2 \right)
$$

Which simplifies to the final computable form:

$$
osc^2_E(f) = |\Omega_E| \int_{\Omega_E} f^2 - \left(\int_{\Omega_E} f\right)^2
$$

For each edge we split the contribution of the edge oscillation into two parts, one for each adjacent cell. This quantity can be computed over a DG0 field using firedrake. 


```
def braess_indicator(self, uh, f, theta=0.5, method="max"):
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
```
