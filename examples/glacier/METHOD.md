# FE method for the SIA

## background

The reference on the transformation of thickness in the shallow ice approximation (SIA) is:

[JB12] G. Jouvet and E. Bueler (2012) _Steady, shallow ice sheets as obstacle problems: Well-posedness and finite element approximation_, SIAM J. Appl. Math 72 (4), 1292--1314

But the idea appears already in:

[R70] P. A. Raviart (1970) _Sur la resolution de certaines equations paraboliques non lineaires_, J. Funct. Analysis 5, 299--328

The reference for the FE mixed method is, for now, just the

[F24] Firedrake [mixed Poisson example](https://www.firedrakeproject.org/demos/poisson_mixed.py.html).

## shallow ice approximation (SIA) model

Starting from the Glen power $n$, we define constants to use as powers:

```math
\begin{align*}
p &= n+1 \\
\omega &= \frac{p-1}{2p} \\
\phi &= \frac{p+1}{2p}
\end{align*}
```

The typical case has $n=3,p=4,\omega=3/8,\phi=5/8$.

For surface elevation $z=s(x,y)$ and bed elevation $z=b(x,y)$, the thickness is $H=s-b$.  Assume an overall constant $\Gamma>0$ and an elevation-independent surface mass balance function $a(x,y)$.

Then the steady, isothermal, non-sliding SIA equation is

```math
-\nabla \cdot \left(\Gamma H^{p+1} |\nabla s|^{p-2} \nabla s\right) = a \qquad (1)
```

This equation is actually the interior condition in a variational inequality (VI) or nonlinear complementarity problem (NCP) subject to the constraint $H\ge 0$, equivalently $s\ge b$.

### power transformation to tilted $p$-Laplacian form

The transformation is

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

Jouvet & Bueler (2012) call $\Phi(u)$ the "tilt" in the $p$-Laplacian form.  Thus

$$|\nabla s|^{p-2} \nabla s = \omega^{p-1} u^{-(p^2-1)/2p} |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u))$$

Note the cancelling powers in this quantity and in the expression for $H^{p+1}$ (Raviart, 1970).

Let $\gamma = \Gamma \omega^{p-1}$.  Now SIA equation $(1)$ becomes

$$-\nabla \cdot \left(\gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u))\right) = a \qquad (2)$$

### abstract obstacle problem

Now define the admissible set

$$\mathcal{K} = \left\{v \in W_0^{1,p}(\Omega) \,:\, v \ge 0\right\}$$

The VI weak form corresponding to $(2)$ is to find $u\in\mathcal{K}$ so that

$$\int_\Omega \gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u)) \cdot \nabla(u-v) \ge \int_\Omega a (u-v) \qquad (3)$$

for all $v\in\mathcal{K}$.  Note that $(3)$ is _not_ the first-order condition for minimizing any functional.

Abstracting $(3)$ gives a problem we call the _tilted $p$-Laplacian obstacle problem_,

$$\int_\Omega \gamma |\nabla u - Z|^{p-2} (\nabla u - Z) \cdot \nabla(u-v) \ge \int_\Omega a (u-v) \qquad (4)$$

for all $v\in\mathcal{K}$.  In $(4)$ the data are the source function $a(x,y)$ and an abstract (_tilt_) vector field $Z(x,y)$.

VI $(4)$ is the first-order condition for minimizing

$$J(u) = \int_\Omega \frac{\gamma}{p} |\nabla u - Z|^p - a u$$

over $\mathcal{K}$ (Jouvet & Bueler, 2012).  Thus $(4)$ modifies the original $p$-Laplacian diffusion equation $-\nabla\cdot(|\nabla u|^{p-2} \nabla u) = a$ to have an obstacle _and_ to make $\nabla u$ want to agree with $Z$, while balancing the source term $a$.  Jouvet & Bueler (2012; Theorem 3.3) show that equivalent problems $(4)$ and $(5)$ are well-posed.

#### Picard iteration

In the SIA $(3)$ the tilt field $Z=\Phi(u)$ is $u$-dependent.  An option, which sort of works, is to Picard iterate on this dependence.  So we may put a Picard iteration wrapper around $(4)$, updating the $Z$ field with each iteration:

```math
\begin{align*}
Z_k &= \Phi(u_{k-1}) \\
\int_\Omega \gamma |\nabla u_k - Z_k|^{p-2} (\nabla u_k - Z_k) \cdot \nabla(u_k-v) &\ge \int_\Omega a (u_k-v)
\end{align*}
```

## mixed FE method

The goal is to build a mass-conserving mixed method for the SIA, in a form which (I hope!) can handle steep bed slopes.  For conservation the function space for the vertically-integrated mass flux $\mathbf{q}$ will be continuous.  The function space for the transformed thickness $u$ will be discontinuous, for stability.  (_And conservation?  There is much I do not understand_.)

This is easiest starting from the "mixed" NCP form of equation $(2)$:

```math
\begin{align*}
\mathbf{q} &= - \gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u)) \qquad (5a) \\
\nabla\cdot \mathbf{q} - a &\ge 0, \quad u \ge 0, \quad u (\nabla\cdot \mathbf{q} - a) = 0 \qquad (5b)
\end{align*}
```

First we need to remove the derivative from $u$, because it will be discontinuous.  The first step is to relate magnitudes in the flux equation, $|\mathbf{q}| = \gamma |\nabla u - \Phi(u)|^{p-1}$, equivalently

$$|\nabla u - \Phi(u)| = \left(\frac{|\mathbf{q}|}{\gamma}\right)^{1/(p-1)}$$

Thus from $(5a)$,

```math
\begin{align*}
\nabla u - \Phi(u) &= -\frac{\mathbf{q}}{\gamma |\nabla u - \Phi(u)|^{p-2}} = -\frac{\gamma^{(p-2)/(p-1)} \mathbf{q}}{\gamma |\mathbf{q}|^{(p-2)/(p-1)}} \\
    &= -\gamma^{-1/(p-1)} |\mathbf{q}|^{-(p-2)/(p-1)} \mathbf{q} \\
    &= - C |\mathbf{q}|^{r-2} \mathbf{q}
\end{align*}
```

where $C=\gamma^{-1/(p-1)}$ and $r=p/(p-1)$  Typically, $r=4/3$.

The new NCP form of $(5)$ is:

```math
\begin{align*}
\nabla u - \Phi(u) &= - C |\mathbf{q}|^{r-2} \mathbf{q} \qquad (6a) \\
\nabla\cdot \mathbf{q} - a \ge 0, \quad u &\ge 0, \quad u (\nabla\cdot \mathbf{q} - a) = 0 \qquad (6b)
\end{align*}
```

We will find its weak form by multiplying by test functions and integrating by parts so as to remove the derivative from $u$.  The new form has different function spaces motivated by the mixed FE method in [F24].

```math
\begin{align*}
\mathcal{K}_m &= \left\{u \in L^p(\Omega)\,:\, u\ge 0 \text{ a.e.}\right\} \\
\mathcal{W}_m &= W^r(div,\Omega) = \left\{\mathbf{w} \in L^r(\Omega)^2 \,:\, \nabla \cdot \mathbf{w} \in L^r(\Omega)\right\}
\end{align*}
```

These spaces are really just a guess.  However, notice that they are sufficient because, by Holder's inequality, and the fact that $p^{-1} + r^{-1} = 1$, those integral expressions below which do not involve $a$ or $\nabla b$ are finite.  (_The expressions which use the data $a$, $\nabla b$ should finite if this data is assumed to be in $L^\infty$?_)

The weak form is a mixed-space VI.  One derives part of it it by multiplying $(6a)$ by a test function $\mathbf{w} \in \mathcal{W}_m$, and integrating by parts to replace $\nabla u \cdot \mathbf{w}$ by $- u \nabla \cdot \mathbf{w}$.  Another portion follows the usual VI derivation from $(6b)$, with a test function $u-v$ where $v\in\mathcal{K}_m$.

Therefore define

```math
\begin{align*}
b\left((u,\mathbf{q}),(v,\mathbf{w})\right) &= \int_\Omega -u \nabla\cdot \mathbf{w} + \left((C |\mathbf{q}|^{r-2} \mathbf{q} - \Phi(u)\right) \cdot \mathbf{w} \\
   &\qquad\quad + (\nabla\cdot \mathbf{q} - a) v
\end{align*}
```

Now the VI problem is to find $(u,\mathbf{q})\in\mathcal{K}_m\times\mathcal{W}_m$ so that

$$\boxed{b\left((u,\mathbf{q}),(u-v,\mathbf{q} - \mathbf{w})\right) \ge 0 \quad \text{ for all }(v,\mathbf{w})\in\mathcal{K}_m\times\mathcal{W}_m} \qquad (7)$$

#### FE spaces

FIXME We will try $DG_0$ for $\mathcal{K}_m$ and $BDM_1$ for $\mathcal{W}_m$.
