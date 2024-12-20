# Thickness transformation for the SIA

### background

The reference on this shallow ice approximation (SIA) transformation is:

G. Jouvet and E. Bueler (2012) _Steady, shallow ice sheets as obstacle problems: Well-posedness and finite element approximation_, SIAM J. Appl. Math 72 (4), 1292--1314

But the idea appears already in:

P. A. Raviart (1970) _Sur la resolution de certaines equations paraboliques non lineaires_, J. Funct. Analysis 5, 299--328

### SIA in original variables

Starting from the Glen power $n$, we set powers
$$\begin{align*}
p&=n+1 \\
\omega&=\frac{p-1}{2p} \\
\phi&=\frac{p+1}{2p}
\end{align*}$$
The typical case has $n=3,p=4,\omega=3/8,\phi=5/8$.

For surface elevation $z=s(x,y)$ and bed elevation $z=b(x,y)$, the thickness is $H=s-b$.  Assume an overall constant $\Gamma>0$ and an elevation-independent surface mass balance function $a(x,y)$.

Then the steady, isothermal, non-sliding SIA is
$$\begin{equation}
-\nabla \cdot \left(\Gamma H^{p+1} |\nabla s|^{p-2} \nabla s\right) = a
\end{equation}$$

### power transformation to tilted $p$-Laplacian form

The transformation is
    $$H = u^\omega$$
The following calculations are key:
$$\begin{align*}
H^{p+1} &= u^{(p^2-1)/2p} \\
\nabla s &= \nabla (H + b) = \omega u^{-\phi} (\nabla u + \omega^{-1} u^\phi \nabla b)
\end{align*}$$
Let
$$\Phi(u) = - \omega^{-1} u^\phi \nabla b$$
Jouvet & Bueler (2012) call $\Phi(u)$ the "tilt" in the $p$-Laplacian form.  Thus
$$|\nabla s|^{p-2} \nabla s = \omega^{p-1} u^{-(p^2-1)/2p} |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u))$$
Note the cancelling powers in this quantity and in the expression for $H^{p+1}$ (Raviart, 1970).  Now SIA equation $(1)$ becomes
$$\begin{equation}
-\nabla \cdot \left(\Gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u))\right) = a
\end{equation}$$

### abstract obstacle problem

Now define the admissible set
$$\mathcal{K} = \{v \in W^{1,p}(\Omega) \,:\, v \ge 0\}$$
The resulting VI weak form is to find $u\in\mathcal{K}$ so that
$$\begin{equation}
\int_\Omega \Gamma |\nabla u - \Phi(u)|^{p-2} (\nabla u - \Phi(u)) \cdot \nabla(u-v) \ge \int_\Omega a (u-v)
\end{equation}$$
for all $v\in\mathcal{K}$.

Abstracting $(3)$ gives a problem we call the _tilted $p$-Laplacian obstacle problem_,
$$\begin{equation}
\int_\Omega \Gamma |\nabla u - Z|^{p-2} (\nabla u - Z) \cdot \nabla(u-v) \ge \int_\Omega a (u-v)
\end{equation}$$
for all $v\in\mathcal{K}$.  In $(4)$ the data are the source function $a(x,y)$ and an abstract (_tilt_) vector field $Z(x,y)$.

VI $(4)$ is the first-order condition for minimizing
$$\begin{equation}
J(u) = \int_\Omega \frac{\Gamma}{p} |\nabla u - Z|^p - a u
\end{equation}$$
over $\mathcal{K}$ (Jouvet & Bueler, 2012).  Thus $(4)$ modifies the original $p$-Laplacian diffusion equation $-\nabla\cdot(|\nabla u|^{p-2} \nabla u) = a$ to have an obstacle _and_ to make $\nabla u$ want to agree with $Z$, while balancing the source term $a$.  Jouvet & Bueler (2012; Theorem 3.3) show that equivalent problems $(4)$ and $(5)$ are well-posed.

### Picard iteration

In the SIA $(3)$ the tilt field $Z=\Phi(u)$ is $u$-dependent.  Note that $(3)$ is _not_ the first-order condition for minimizing any functional.

To solve the transformed SIA obstacle problem $(3)$ we put a Picard iteration wrapper around $(4)$:
$$\begin{align*}
Z_k &= \Phi(u_{k-1}) \\
\int_\Omega \Gamma |\nabla u_k - Z_k|^{p-2} (\nabla u_k - Z_k) \cdot \nabla(u_k-v) &\ge \int_\Omega a (u_k-v)
\end{align*}$$
