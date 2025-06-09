# FE method for the glacier problem

## introduction

This `.md` documents aspects of `steady.py` and its numerical methods.  We suppose that our mathematical glacier problem determines the steady-state surface elevation, equivalently thickness, of a glacier in a fixed time-independent climate and on a fixed bedrock topography.  The ice flow model used here is the shallow ice approximation (SIA).  Note that a power transformation of the ice thickness will be applied when solving the SIA; see below.

## shallow ice approximation model

Assume a bed elevation function $z=b(x,y)$.  Either the surface elevation $z=s(x,y)$ or the corresponding thickness $H(x,y)=s(x,y)-b(x,y)$ is the primary unknown of the model.

We also assume, as data, a surface mass balance function $a(x,y,s)$ which might depend on surface elevation.

Assume an overall flow constant $\Gamma>0$.  Starting from the Glen power $n\ge 1$, we define constants for use as powers:

```math
\begin{align*}
p &= n+1 \\
\omega &= \frac{p-1}{2p} \\
\phi &= \frac{p+1}{2p}
\end{align*}
```

The typical/standard case for glaciology has $n=3,p=4,\omega=3/8,\phi=5/8$.

The steady, isothermal, non-sliding SIA equation is

```math
-\nabla \cdot \left(\Gamma H^{p+1} |\nabla s|^{p-2} \nabla s\right) = a(s) \qquad (1)
```

Note that dependence on $x,y$ will generally be suppressed from here on.

This equation is actually the interior condition in a variational inequality (VI) or nonlinear complementarity problem (NCP) subject to the constraint $H\ge 0$ [JB12], equivalently $s\ge b$.

### power transformation

The following transformation, introduced in [R70] and applied to non-flat beds in [JB12], converts the SIA equation (1) to a version of the better-known $p$-Laplacian equation [BL93]:

$$H = u^\omega$$
Here $\omega = (p-1)/(2p)$ as above.  The following calculations allow us to transform parts of (1):

```math
\begin{align*}
H^{p+1} &= u^{(p^2-1)/2p} \\
\nabla s &= \nabla (H + b) = \omega u^{-\phi} (\nabla u + \omega^{-1} u^\phi \nabla b)
\end{align*}
```

Let

$$\Phi(u) = - \omega^{-1} u^\phi \nabla b$$

Note that [JB12] calls $\Phi(u)$ the "tilt" because it arises from the bed elevation gradient.  Thus

$$|\nabla s|^{p-2} \nabla s = \omega^{p-1} u^{-(p^2-1)/2p} |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u))$$

Note the sign of the powers in this quantity, and in the expression for $H^{p+1}$.  Let $\gamma = \Gamma \omega^{p-1}$.  Now SIA equation $(1)$ becomes

$$-\nabla \cdot \left(\gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u))\right) = a(u^\omega+b) \qquad (2)$$

Equation $(2)$ is a modified form of the $p$-Laplacian equation.  Generally we are in the degenerate $p>2$ case.  There does not seem to be a literature which really helps us understand understand the effect of the tilt $\Phi(u)$, but see commentary in [JB12].  Regarding consequences of the dependence of $a$ on the surface elevation, and an understanding of such problems as energy minimization, at least when the bed is flat, see [JRBB11].

### obstacle problem in transformed thickness

Now define the admissible set

$$\mathcal{K} = \left\{v \in W_0^{1,p}(\Omega) \,:\, v \ge 0\right\}$$

The VI weak form corresponding to $(2)$ is to find $u\in\mathcal{K}$ so that

$$\int_\Omega \gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u)) \cdot \nabla(u-v) \ge \int_\Omega a(u^\omega+b) (u-v) \qquad (3)$$

for all $v\in\mathcal{K}$ [JB12].  VI (3) is derived, as usual, by multiplying $(2)$ by the test function $u-v$, and integrating by parts.

Our primary goal is an accurate finite element (FE) solution of problem $(3)$.  However, there are several approaches to doing this, and it is not a settled matter of which works best.

The most direct method of solving $(3)$ by FE is to use $u\in CG_1$, and application of a Newton iteration to handle the several nonlinearities.  This primal method is used here, with an optional version applying a Picard iteration as in [JB12].  These primal methods often work acceptably, but they can become fragile, especially with regard to Newton convergence when the bed elevation has large gradient [B16].  In the same cases they seem to make mass-conservation errors, essentially of the type discussed in [JAS13], and mitigated by the FE space choices in [B23].  The FE methods considered below only partially addresss these fragility and conservation issues.

### tilted $p$-Laplacian objective

Supposing $\nabla b\ne 0$, then VI $(3)$ is _not_ the first-order condition for minimizing any functional.  We can abstract and simplify $(3)$, to give the _tilted $p$-Laplacian obstacle problem_ [JB12].  Let $Z(x,y)$ be a fixed vector field and $A(x,y)$ a fixed source function.  Consider:

$$\int_\Omega \gamma |\nabla u - Z|^{p-2} (\nabla u - Z) \cdot \nabla(u-v) \ge \int_\Omega A (u-v) \qquad (4)$$

for all $v\in\mathcal{K}$.  In $(4)$ the data are the source function $A$ and the tilt $Z$.

VI $(4)$ is the first-order condition for minimizing

$$J(u) = \int_\Omega \frac{\gamma}{p} |\nabla u - Z|^p - A u$$

over $\mathcal{K}$ [JB12].  Thus $(4)$ modifies a standard $p$-Laplacian diffusion equation $-\nabla\cdot(|\nabla u|^{p-2} \nabla u) = f$ [BL93] to have a zero-function obstacle, and to make $\nabla u$ want to agree with $Z$, while balancing the source term $A$.

Theorem 3.3 in [JB12] shows that the VI problem $(4)$ and the minimization of $J(u)$ are equivalent and well-posed.  However, the SIA VI problem $(3)$ is not in form $(4)$ primarily because of the solution-dependent vector field $Z=\Phi(u)$.  Only existence is known for $(3)$ (even when $a$ is independent of $s$) [JB12].

Elevation-dependent surface mass balance can generally be included into the above minimization framework.  However, for some dependences $a(s)$, non-existence is known [JRBB11] (even on a flat bed).

### Picard iteration

One way to solve VI problem $(3)$ is to directly apply a VI-adapted Newton solver, specifically the `vinewtonrsls` SNES type from PETSc.  This works in many circumstances, as long as the initial iterate is nonzero in a roughly reasonable pattern, and as long as grid sequencing is used.

In the VI problem $(3)$ the tilt field $Z=\Phi(u)$ is in fact $u$-dependent.  Likewise $A$ may be $u$-dependent.  An option for solving the SIA with these dependencies is to Picard iterate on solving $(4)$ [JB12].  That is, we may put an iterative wrapper around $(4)$, updating the $Z$ field, and the $A$ source, with each iteration.  That is, we find $u_k \in \mathcal{K}$ so that, for all $v\in\mathcal{K}$,

```math
\begin{align*}
Z_k &= \Phi(u_{k-1}), \quad A_k = a(u_{k-1}^\omega +b), \\
\int_\Omega \gamma |\nabla u_k - Z_k|^{p-2} &(\nabla u_k - Z_k) \cdot \nabla(u_k-v) \ge \int_\Omega A_k (u_k-v) \qquad (5)
\end{align*}
```

This solution method seems to converge well when solved as a primal problem using FE methods.  A stopping criterion for Picard iterations is not clear, and we simply use a fixed number.  Note that solving $(5)$ still requires a VI-adapted Newton solver; it is an obstacle problem with a nonlinear operator, but at least it is close to $p$-Laplacian type, versus solving $(3)$ directly.

However, with $u \in CG_1$ in this primal form, mass conservation issues remain when and where $\nabla b$ is large in magnitude [B23, JAS13].  Fixing this is a longer-term project.

### References

[BL93] Barrett, J. W. and Liu, W. B. (1993). _Finite element approximation of the $p$-Laplacian_. Mathematics of Computation 61 (204), 523--537

[B23] Brinkerhoff, D. J. (2023). _Compatible finite elements for glacier modeling_. Computing in Science & Engineering, 25 (3), 18--28

[B16] Bueler, E. (2016). _Stable finite volume element schemes for the shallow-ice approximation_. Journal of Glaciology, 62 (232), 230--242

[JAS13] Jarosch, A. H., Schoof, C. G., and Anslow, F. S. (2013). _Restoring mass conservation to shallow ice flow models over complex terrain_, The Cryosphere 7 (1), 229--240

[JB12] G. Jouvet and E. Bueler (2012). _Steady, shallow ice sheets as obstacle problems: Well-posedness and finite element approximation_, SIAM J. Appl. Math 72 (4), 1292--1314

[JRBB11] G. Jouvet, J. Rappaz, E. Bueler, and H. Blatter (2011). _Existence and stability of steady-state solutions of the shallow-ice-sheet equation by an energy-minimization approach_, Journal of Glaciology, 57 (202), 345--354

[R70] P. A. Raviart (1970). _Sur la resolution de certaines equations paraboliques non lineaires_, J. Funct. Analysis 5, 299--328
