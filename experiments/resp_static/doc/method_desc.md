## Goal

We consider a network of $N$ interconnected neural populations. Each population receives recurrent input from the network, including itself, and an external input $h_n$. We write the firing rates and external inputs as column vectors $\bar{r}=(r_1,\ldots,r_N)^T$ and $\bar{h}=(h_1,\ldots,h_N)^T$.

If the network settles under a constant $\bar{h}$, we can view its steady-state response as a map $S\colon \bar{h}\mapsto\bar{r}$. A simulator can evaluate $S$ even when we have no explicit expression for it. An exhaustive grid with $m$ values per input requires $m^N$ simulations. Our goal is to approximate the map near a chosen operating point with a practical number of evaluations.

## Motivation

Suppose the network dynamics can be written in rate form,

$$
\tau \frac{d r_n}{dt} = -r_n + f_n(\bar{r},h_n),
\qquad n=1,\ldots,N.
$$

Let $F(\bar{r},\bar{h})=(f_1(\bar{r},h_1),\ldots,f_N(\bar{r},h_N))^T$. For example, a Wilson–Cowan-like rate model can use $F(\bar{r},\bar{h})=g(W\bar{r}+\bar{h})$, with a recurrent weight matrix $W$ and an elementwise gain function $g$. Every steady state satisfies the self-consistency equation

$$
S(\bar{h}) = F(S(\bar{h}),\bar{h}).
$$

We approximate the local response to an input perturbation by expanding this equation at a baseline fixed point $(\bar{r}_0,\bar{h}_0)$, where $\bar{r}_0=S(\bar{h}_0)$.

## Baseline fixed point

To estimate $S(\bar{h})$ from evaluations of $F$, we start from $\bar{r}^{(0)}$ and apply a relaxed fixed-point iteration,

$$
\bar{r}^{(t+1)}
= (1-\alpha)\bar{r}^{(t)}
+ \alpha F(\bar{r}^{(t)},\bar{h}),
\qquad 0<\alpha\leq 1.
$$

The relaxation factor $\alpha$ sets how far each iterate moves toward $F(\bar{r}^{(t)},\bar{h})$. We assess convergence using the fixed-point residual $\|F(\bar{r}^{(t)},\bar{h})-\bar{r}^{(t)}\|$. The starting rate can affect which fixed point the iteration reaches when several exist.

## Linear response

Let $J=D_{\bar{r}}F(\bar{r}_0,\bar{h}_0)$ and $Q=D_{\bar{h}}F(\bar{r}_0,\bar{h}_0)$. The first-order expansion of the fixed-point equation gives

$$
\Delta\bar{r} \approx J\Delta\bar{r} + Q\Delta\bar{h}.
$$

If $I-J$ is invertible, the predicted perturbed rate is

$$
\hat{\bar{r}}
= \bar{r}_0 + (I-J)^{-1}Q\Delta\bar{h}.
$$

Here $Q$ describes the direct effect of the external input, while solving the linear system with $I-J$ accounts for recurrent feedback. The same matrices can be reused for multiple input perturbations around the baseline.

## Finite-difference estimates

When the derivatives are unavailable, we estimate them by perturbing one coordinate at a time. Let $\bar{e}_n$ be the $n$th coordinate vector, and let $\epsilon_r$ and $\epsilon_h$ be the rate and input step sizes. The forward differences are

$$
\begin{aligned}
J_{kn} &\approx
\frac{F_k(\bar{r}_0+\epsilon_r\bar{e}_n,\bar{h}_0)
      -F_k(\bar{r}_0,\bar{h}_0)}{\epsilon_r},\\
Q_{kn} &\approx
\frac{F_k(\bar{r}_0,\bar{h}_0+\epsilon_h\bar{e}_n)
      -F_k(\bar{r}_0,\bar{h}_0)}{\epsilon_h}.
\end{aligned}
$$

For the model above, the instantaneous response of population $k$ depends directly only on $h_k$, so $Q$ is diagonal. The finite differences require the ability to evaluate $F$ with independently specified rates and inputs. A simulator that returns only steady states provides evaluations of $S$.

## Second-order response

Retaining the quadratic terms in the Taylor expansion of $f_k$ gives an implicit equation for the predicted rate change $\bar{x}=\hat{\bar{r}}-\bar{r}_0$,

$$
\big[(I-J)\bar{x}\big]_k =
(Q\Delta\bar{h})_k
+ \tfrac12 \bar{x}^{T} H_k \bar{x}
+ \Delta h_k (M\bar{x})_k
+ \tfrac12 q_k(\Delta h_k)^2.
$$

At the baseline, $(H_k)_{mn}=\partial_{r_m}\partial_{r_n}f_k$, $M_{km}=\partial_{h_k}\partial_{r_m}f_k$, and $q_k=\partial_{h_k}^2 f_k$. These derivatives describe curvature with respect to recurrent rates, the change in recurrent sensitivity with external input, and curvature with respect to external input, respectively. Each $f_k$ depends directly on $h_k$ alone, so the input terms in the $k$th equation involve only $\Delta h_k$.

### Centered finite differences

Write $\phi_k(\bar{a},b)=f_k(\bar{r}_0+\bar{a},h_{0,k}+b)$ for the response of population $k$ to rate and input offsets, and let $\bar{u}_m=\epsilon_r\bar{e}_m$. We estimate $J$ and the diagonal entries of $Q$ for the second-order model using centered first differences along the rate and input coordinates. For the rate Hessian we evaluate all four sign combinations of the two rate offsets,

$$
(H_k)_{mn}\approx
\frac{\phi_k(\bar{u}_m+\bar{u}_n,0)
-\phi_k(\bar{u}_m-\bar{u}_n,0)
-\phi_k(-\bar{u}_m+\bar{u}_n,0)
+\phi_k(-\bar{u}_m-\bar{u}_n,0)}{4\epsilon_r^2}.
$$

For $m=n$, this stencil evaluates the rate offsets $-2\bar{u}_m$, $\bar{0}$, and $2\bar{u}_m$. The input curvature follows from a centered second difference at fixed rates,

$$
q_k\approx
\frac{\phi_k(\bar{0},\epsilon_h)-2\phi_k(\bar{0},0)
+\phi_k(\bar{0},-\epsilon_h)}{\epsilon_h^2}.
$$

For the mixed derivative, we perturb one rate coordinate and the external input to population $k$,

$$
M_{km}\approx
\frac{\phi_k(\bar{u}_m,\epsilon_h)-\phi_k(-\bar{u}_m,\epsilon_h)
-\phi_k(\bar{u}_m,-\epsilon_h)+\phi_k(-\bar{u}_m,-\epsilon_h)}
{4\epsilon_r\epsilon_h}.
$$

### Newton iteration

For a fixed $\Delta\bar{h}$, let $b_k=(Q\Delta\bar{h})_k+\tfrac12 q_k(\Delta h_k)^2$. We initialize the rate change by solving $(I-J)\bar{x}^{(0)}=\bar{b}$, which retains the input curvature. The residual of the quadratic equation and its Jacobian are

$$
\begin{aligned}
g_k(\bar{x}) &= \big[(I-J)\bar{x}\big]_k
-\Delta h_k(M\bar{x})_k-\tfrac12\bar{x}^{T}H_k\bar{x}-b_k,\\
A_{km}(\bar{x}) &= (I-J)_{km}-\Delta h_k M_{km}-(H_k\bar{x})_m,
\end{aligned}
$$

where $H_k$ is taken to be symmetric. At each iteration we solve $A(\bar{x})\bar{\delta}=-\bar{g}(\bar{x})$ and update $\bar{x}$ by $\lambda\bar{\delta}$. Backtracking reduces $\lambda$ when the trial step fails to decrease the residual norm. We stop when $\|\bar{g}(\bar{x})\|\leq \mathrm{tol}(1+\|\bar{b}\|)$ or the iteration limit is reached, and take $\hat{\bar{r}}=\bar{r}_0+\bar{x}$.

## Validation

For each test perturbation $\Delta\bar{h}$, we compare the linear and second-order predictions with a fixed point recomputed at $\bar{h}_0+\Delta\bar{h}$. Varying the perturbation magnitude and sampling multiple directions tests how far the local approximations remain accurate. We can summarize errors across populations using absolute deviations or, when the reference rates are safely away from zero, relative deviations, then report their distribution across trials.
