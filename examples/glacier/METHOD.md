# FE method for the glacier problem

## introduction

Our glacier problem is to determine the steady-state surface elevation, equivalently thickness, of a glacier in a fixed (time-dependent) climate.  The ice flow model used here is the shallow ice approximation (SIA).  References on the transformation of thickness in the SIA, used below, are [JB12] and [R70].

This `.md` file documents aspects of `steady.py` and its numerical methods.  The reference for the FE mixed method is, for now, only [F24].

## shallow ice approximation model

Assume an overall flow constant $\Gamma>0$, an elevation-independent surface mass balance function $a(x,y)$, and a bed elevation function $z=b(x,y)$.  Starting from the Glen power $n$, we define constants for use as powers:

```math
\begin{align*}
p &= n+1 \\
\omega &= \frac{p-1}{2p} \\
\phi &= \frac{p+1}{2p}
\end{align*}
```

(The typical case has $n=3,p=4,\omega=3/8,\phi=5/8$.)  For surface elevation $z=s(x,y)$ the thickness is $H=s-b$.  Then the steady, isothermal, non-sliding SIA equation, in the original variables, is

```math
-\nabla \cdot \left(\Gamma H^{p+1} |\nabla s|^{p-2} \nabla s\right) = a \qquad (1)
```

This equation is actually the interior condition in a variational inequality (VI) or nonlinear complementarity problem (NCP) subject to the constraint $H\ge 0$ [JB12], equivalently $s\ge b$.  This will be shown below in transformed variables.

### power transformation

The following transformation, introduced in [R70] and applied to non-flat beds in [JB12], converts the SIA equation to a form of the better-known $p$-Laplacian equation [BL93]:

$$H = u^\omega$$
The following calculations are key:

```math
\begin{align*}
H^{p+1} &= u^{(p^2-1)/2p} \\
\nabla s &= \nabla (H + b) = \omega u^{-\phi} (\nabla u + \omega^{-1} u^\phi \nabla b)
\end{align*}
```

Let

$$\Phi(u) = - \omega^{-1} u^\phi \nabla b$$

([JB12] suggests calling $\Phi(u)$ the "tilt" because it arises from the bed elevation gradient.)  Thus

$$|\nabla s|^{p-2} \nabla s = \omega^{p-1} u^{-(p^2-1)/2p} |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u))$$

Note the opposite-sign powers in this quantity and in the expression for $H^{p+1}$, which is the point.  Let $\gamma = \Gamma \omega^{p-1}$.

Now SIA equation $(1)$ becomes

$$-\nabla \cdot \left(\gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u))\right) = a \qquad (2)$$

Equation $(2)$ is a modified form of the $p$-Laplacian equation, in the degenerate $p>2$ case.  There does not seem to be a literature which helps us understand the tilt $\Phi(u)$, but see [JB12].

### obstacle problem

Now define the admissible set

$$\mathcal{K} = \left\{v \in W_0^{1,p}(\Omega) \,:\, v \ge 0\right\}$$

The VI weak form corresponding to $(2)$ is to find $u\in\mathcal{K}$ so that

$$\int_\Omega \gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u)) \cdot \nabla(u-v) \ge \int_\Omega a (u-v) \qquad (3)$$

for all $v\in\mathcal{K}$ [JB12].  It is derived, as usual, by multiplying $(2)$ by the test function $u-v$, and integrating by parts.

Our primary goal is an accurate finite element (FE) solution of problem $(2)$.  However, there are several approaches to doing this, and it is not a settled matter of which works best.  Further transformations of the continuum problem will be needed to discuss the various FE methods.

The most direct method of solving $(3)$ by an FE method is to use $u\in CG_1$ and apply a Newton iteration to handle the nonlinearity.  This often works, but it becomes very fragile, with regard to Newton convergence, when the bed elevation has large gradient.  In the same case it seems to make bad mass-conservation errors, of the type discussed in [JAS13].  The goals of the different/novel FE methods considered below are to address the fragility and conservation issues.

### tilted $p$-Laplacian form, and Picard iteration

Supposing $\nabla b\ne 0$, then VI $(3)$ is _not_ the first-order condition for minimizing any functional.  We can abstract and simplify $(3)$, to give the _tilted $p$-Laplacian obstacle problem_ [JB12].  Let $Z(x,y)$ be a fixed vector field.  Consider:

$$\int_\Omega \gamma |\nabla u - Z|^{p-2} (\nabla u - Z) \cdot \nabla(u-v) \ge \int_\Omega a (u-v) \qquad (4)$$

for all $v\in\mathcal{K}$.  In $(4)$ the data are the source function $a$ and the tilt $Z$.

VI $(4)$ is the first-order condition for minimizing

$$J(u) = \int_\Omega \frac{\gamma}{p} |\nabla u - Z|^p - a u$$

over $\mathcal{K}$ [JB12].  Thus $(4)$ modifies the original $p$-Laplacian diffusion equation $-\nabla\cdot(|\nabla u|^{p-2} \nabla u) = a$ [BL93] to have a zero-function obstacle, and to make $\nabla u$ want to agree with $Z$, while balancing the source term $a$.

Theorem 3.3 in [JB12] shows that the VI problem $(4)$ and the minimization of $J(u)$ are equivalent and well-posed.  However, VI problem $(3)$ is not in form $(4)$ because of the solution-dependent vector field $Z=\Phi(u)$.  Only existence is known for $(3)$ [JB12].

In the VI problem $(3)$ the tilt field $Z=\Phi(u)$ is $u$-dependent.  Thus an option for solving the SIA, which sort-of works, is to Picard iterate on solving $(4)$, with this dependence [JB12].  That is, we may put an iterative wrapper around $(4)$, updating the $Z$ field with each iteration:

```math
\begin{align*}
Z_k &= \Phi(u_{k-1}) \\
\int_\Omega \gamma |\nabla u_k - Z_k|^{p-2} (\nabla u_k - Z_k) \cdot \nabla(u_k-v) &\ge \int_\Omega a (u_k-v)
\end{align*}
```

This method of solution seems to generally converge when solved as a primal problem using FE methods.  However, with $u \in CG_1$, i.e. in primal form, the mass conservation issues remain when and where $\nabla b$ is large in magnitude.

### References

[BL93] Barrett, J. W. and Liu, W. B. (1993). _Finite element approximation of the $p$-Laplacian_. Mathematics of Computation 61(204), 523--537

[F24] Firedrake mixed Poisson example at [demos/poisson_mixed.py.html](https://www.firedrakeproject.org/demos/poisson_mixed.py.html)

[JAS13] Jarosch, A. H., Schoof, C. G., and Anslow, F. S. (2013). _Restoring mass conservation to shallow ice flow models over complex terrain_, The Cryosphere 7 (1), 229--240

[JB12] G. Jouvet and E. Bueler (2012) _Steady, shallow ice sheets as obstacle problems: Well-posedness and finite element approximation_, SIAM J. Appl. Math 72 (4), 1292--1314

[R70] P. A. Raviart (1970) _Sur la resolution de certaines equations paraboliques non lineaires_, J. Funct. Analysis 5, 299--328
