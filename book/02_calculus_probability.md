# 6 · Derivatives and optimisation in one variable

## 6.1 Three readings of one number

The derivative f′(x) is the limit of the difference quotient, and it is
simultaneously the instantaneous rate of change, the slope of the tangent, and
the coefficient of the best linear approximation at x *[MML p.343]*. For
optimisation there is a fourth reading, and it is the one that matters:
**f′(x) is a one-dimensional vector pointing towards increase** *[MML p.382]*.
Positive means "uphill to the right", negative means "uphill to the left", and
the magnitude is the steepness.

Hold on to that, because gradient descent is nothing more than taking that
sentence seriously in n dimensions.

## 6.2 Differentiating numerically

You will often need a derivative you have not derived — to check an analytic
gradient, or to probe a black box. The forward difference is the obvious
choice and the wrong one. Taylor tells you why:

<div class="eq">f(x+h) = f(x) + hf′(x) + (h²⁄2)f″(x) + O(h³)</div>

Rearranging gives the forward difference with an error of (h/2)f″(x) = **O(h)**.
Now expand in both directions and subtract:

<div class="eq">f(x+h) − f(x−h) = 2hf′(x) + (h³⁄3)f‴(x) + O(h⁵)</div>

<div class="eq">(f(x+h) − f(x−h))⁄2h = f′(x) + (h²⁄6)f‴(x) + O(h⁴)</div>

The second-order term **cancels**, leaving error **O(h²)**. One extra function
evaluation buys a whole order of accuracy — that is why every gradient check
in deep learning uses central differences.

### The error floor, worked out

Smaller h is not monotonically better, and this trips up almost everyone the
first time. Two errors move in opposite directions:

- **truncation** falls as h shrinks: ≈ (h²/6)|f‴|;
- **round-off** *grows* as h shrinks: the numerator is a difference of two
  nearly equal floats, each carrying absolute error ≈ ε|f|, and you then
  divide by 2h, giving ≈ ε|f|/h.

Total error ≈ (h²/6)|f‴| + ε|f|/h. Differentiate with respect to h and set to
zero:

<div class="eq">(h⁄3)|f‴| − ε|f|⁄h² = 0 &nbsp;⟹&nbsp; h³ = 3ε|f|⁄|f‴| &nbsp;⟹&nbsp; h ∝ ε<sup>1/3</sup></div>

With ε ≈ 2.2×10⁻¹⁶, that is **h ≈ 6×10⁻⁶** — not the smallest representable
number, and the resulting error floor is around 10⁻¹¹, not 10⁻¹⁶.

```python
"""The U-shaped error curve of numerical differentiation."""

import numpy as np


def central_difference(f, x: float, h: float) -> float:
    """Approximates f'(x) by a central difference.

    Args:
        f: Scalar function of one variable.
        x: Point of evaluation.
        h: Step size.

    Returns:
        The approximate derivative.
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


f = np.sin
exact = np.cos(1.0)
print(f'{"h":>10}  {"estimate":>18}  {"abs error":>12}')
for power in range(1, 16):
    h = 10.0 ** (-power)
    got = central_difference(f, 1.0, h)
    print(f'{h:>10.0e}  {got:>18.12f}  {abs(got - exact):>12.3e}')
```

Run it and read the column: the error falls, bottoms out near 1e-11 around
h = 1e-5 or 1e-6, then *climbs again* — exactly where the algebra above said
it would.

## 6.3 Gradient descent

Minimising by solving f′(x) = 0 fails in practice for two reasons: the
stationary points include maxima and inflections, and the equation is
unsolvable except in trivial cases — linear regression with squared error
being the notable exception *[MML p.381]*. So we iterate instead.

To *maximise*, step along the derivative. To **minimise**, step against it
*[MML p.383–384]*:

<div class="eq">x<sub>n+1</sub> := x<sub>n</sub> − h·f′(x<sub>n</sub>)</div>

where h > 0 is the **learning rate**. That is the entire algorithm, and every
optimiser in Part III is a variation on it.

### Why a step downhill reduces the loss: the descent lemma

The mountain analogy says "downhill is good" but not "how far". Taylor with a
remainder does. Suppose the gradient is L-Lipschitz, meaning
|f′(a) − f′(b)| ≤ L|a − b| — curvature is bounded by L. Then

<div class="eq">f(y) ≤ f(x) + f′(x)(y − x) + (L⁄2)(y − x)²</div>

Substitute the gradient step y = x − hf′(x):

<div class="eq">f(x − hf′(x)) ≤ f(x) − h·f′(x)² + (L⁄2)h²f′(x)² = f(x) − h(1 − Lh⁄2)·f′(x)²</div>

The loss *decreases* whenever h(1 − Lh/2) > 0, that is whenever

<div class="eq">0 &lt; h &lt; 2⁄L</div>

and the guaranteed decrease is largest at h = 1/L. ∎

**This is the entire theory of learning rates in two lines.** It also tells
you what L means operationally: the largest curvature your loss has anywhere
you travel.

### The same result on a quadratic, exactly

Let f(x) = (a/2)x² with a > 0, so f′(x) = ax and L = a. Then

<div class="eq">x<sub>n+1</sub> = x<sub>n</sub> − h·a·x<sub>n</sub> = (1 − ha)·x<sub>n</sub> &nbsp;⟹&nbsp; x<sub>n</sub> = (1 − ha)<sup>n</sup>·x₀</div>

Convergence iff |1 − ha| < 1, i.e. 0 < h < 2/a — matching the lemma, now with
no inequality slack. Four regimes, which you should be able to recognise from
a loss curve alone:

| Learning rate | Factor (1 − ha) | Behaviour |
|---|---|---|
| h < 1/a | in (0, 1) | smooth monotone convergence |
| h = 1/a | 0 | converges in **one step** |
| 1/a < h < 2/a | in (−1, 0) | converges, oscillating in sign |
| h = 2/a | −1 | perpetual oscillation, no progress |
| h > 2/a | < −1 | **divergence**, growing geometrically |

### Worked iterates

Take a = 2 (so L = 2, the limit is h = 1) and x₀ = 1. Three learning rates,
five steps each:

| n | h = 0.25 | h = 0.5 (=1/a) | h = 1.1 |
|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.000 |
| 1 | 0.500 | 0.000 | −1.200 |
| 2 | 0.250 | 0.000 | 1.440 |
| 3 | 0.125 | 0.000 | −1.728 |
| 4 | 0.0625 | 0.000 | 2.074 |
| 5 | 0.03125 | 0.000 | −2.488 |

Halving per step; exact in one step; and alternating while growing by 1.2× per
step. The third column is divergence — and notice it *oscillates*, which is
the practical signature: a loss that zigzags upward means the step is past
2/L, not that the data is bad.

### In many dimensions

The Hessian replaces a. Diagonalise it (it is symmetric — Chapter 5): along
eigenvector i the iteration contracts by (1 − hλᵢ). So:

- **stability** is set by the *largest* eigenvalue: h < 2/λ_max;
- **speed** is set by the *smallest*: the slowest direction contracts by
  (1 − hλ_min) per step.

With h just under 2/λ_max, the slow direction's factor is
1 − 2λ_min/λ_max = 1 − 2/κ. **The number of iterations scales with the
condition number κ.** That is the zig-zag you have seen in contour plots, and
it is why feature scaling is not cosmetic: it is conditioning the Hessian.

```python
"""Gradient descent, and the stability limit h < 2/L."""

import numpy as np


def gradient_descent(grad, x0: np.ndarray, lr: float, iters: int,
                     momentum: float = 0.0) -> np.ndarray:
    """Minimises a function given its gradient.

    Divergence above the stability limit is a real outcome and is not
    clamped away: a routine that cannot diverge cannot teach you where
    the limit is.

    Args:
        grad: Function returning the gradient at a point.
        x0: Starting point.
        lr: Learning rate.
        iters: Number of steps.
        momentum: Heavy-ball coefficient; 0.0 is plain descent.

    Returns:
        The final iterate.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    velocity = np.zeros_like(x)
    for _ in range(iters):
        velocity = momentum * velocity - lr * np.asarray(grad(x), dtype=float)
        x = x + velocity
        if not np.all(np.isfinite(x)):
            break
    return x


curvature = np.array([4.0, 1.0])            # L = 2 * max(curvature) = 8
limit = 2.0 / (2.0 * curvature.max())       # stability limit on h
grad = lambda x: 2.0 * curvature * x        # f(x) = Σ aᵢxᵢ²

for scale in (0.1, 0.5, 0.99, 1.01, 1.5):
    lr = scale * limit
    final = gradient_descent(grad, np.array([5.0, -3.0]), lr, 500)
    size = np.linalg.norm(final)
    verdict = 'converged' if np.isfinite(size) and size < 1e-6 else (
        'DIVERGED' if not np.isfinite(size) or size > 1e3 else 'crawling')
    print(f'h = {lr:7.4f} ({scale:4.2f} x limit)   ‖x‖ = {size:9.3e}   {verdict}')
```

### Gradient descent is a discretised ODE

There is a deeper reading *[MML p.389]*. Consider the **gradient flow**

<div class="eq">x′(t) = −f′(x(t))</div>

a trajectory that always moves downhill. Its forward-Euler discretisation with
step h is x_{n+1} = x_n − h f′(x_n) — gradient descent, exactly. So the
iteration is a numerical approximation of a continuous descent, and everything
known about unstable step sizes in numerical integration applies to learning
rates. The divergence above 2/L is the same instability that makes Euler blow
up on a stiff ODE, and "stiffness" is just another word for a large condition
number.

### Momentum

Add memory: v_{n+1} = βv_n − h∇f, x_{n+1} = x_n + v_{n+1}. On a quadratic, a
component with curvature λ now follows a *second-order* recurrence whose
behaviour is governed by the roots of

<div class="eq">z² − (1 + β − hλ)z + β = 0</div>

Oscillating components (which flip sign every step) cancel across steps while
consistent ones accumulate — the effective step along a persistent direction
is multiplied by about 1/(1 − β). At β = 0.9 that is a tenfold amplification
of the slow direction, which is why momentum so often turns a crawl into a
sprint on ill-conditioned problems.

## 6.4 Gate for this chapter

| Check | Threshold |
|---|---|
| Central-difference error curve | minimum near h ≈ 1e-5, then rises |
| Your worked iterate table at a = 2 | matches to 1e-12 |
| Convergence for h < 2/L | ‖x‖ < 1e-6 |
| **Divergence for h > 2/L** | required observation, recorded |
| Momentum vs plain on curvature ratio 4:1 | fewer iterations to tolerance |

---

# 7 · Many variables: gradients, Jacobians, backpropagation

## 7.1 Gradient

For f : ℝⁿ → ℝ, the **partial derivative** ∂f/∂xᵢ differentiates in one
coordinate holding the rest fixed, and the **gradient** collects them
*[MML p.448]*:

<div class="eq">∇f(<b>x</b>) = (∂f⁄∂x₁, …, ∂f⁄∂x<sub>n</sub>)</div>

### Why the gradient is the steepest direction

The directional derivative along a unit vector **u** is

<div class="eq">D<sub><b>u</b></sub>f = lim<sub>t→0</sub> (f(<b>x</b> + t<b>u</b>) − f(<b>x</b>))⁄t = ⟨∇f, <b>u</b>⟩</div>

By Cauchy–Schwarz, ⟨∇f, **u**⟩ ≤ ‖∇f‖·‖**u**‖ = ‖∇f‖, with equality precisely
when **u** points along ∇f. So the gradient is the direction of **steepest
ascent**, its negative the steepest descent, and its norm is the steepness. ∎

That single inequality is what licenses Chapter 6's algorithm in n dimensions
*[MML p.481]*.

## 7.2 Jacobian and the chain rule

For a vector-valued f : ℝⁿ → ℝᵐ, the derivative is the **Jacobian**, the m×n
matrix of all partials *[MML p.460]*. The chain rule then becomes matrix
multiplication:

<div class="eq">J<sub>f∘g</sub>(<b>x</b>) = J<sub>f</sub>(g(<b>x</b>)) · J<sub>g</sub>(<b>x</b>)</div>

Read that line twice, because **it is backpropagation**. A neural network is a
composition f_L ∘ ⋯ ∘ f_1; the gradient of the loss is a product of Jacobians;
and the only question is the order in which you multiply them.

### The cost argument for reverse mode

Consider a chain ℝⁿ → ℝ^{h} → ℝ^{h} → ℝ, with Jacobians J₃ (1×h), J₂ (h×h),
J₁ (h×n). The product J₃J₂J₁ can be bracketed two ways:

| Order | Operations | Intermediate |
|---|---|---|
| (J₃J₂)J₁ — **reverse** | h² + hn | a 1×h row vector |
| J₃(J₂J₁) — forward | h²n + hn | an h×n matrix |

Reverse mode is cheaper by a factor of about n — the number of inputs. Since a
loss has **one output** and millions of inputs, reverse mode wins
overwhelmingly, and reverse mode *is* backpropagation. The cost of a gradient
is then a small constant times the cost of a forward pass, independently of
the parameter count. **Every deep model in existence rests on that
asymmetry.**

## 7.3 Backpropagation, derived in full

Do this once by hand and autograd stops being magic. Two layers, scalar
output, squared error:

<div class="eq">z₁ = W₁<b>x</b> + <b>b</b>₁ &nbsp;·&nbsp; a₁ = σ(z₁) &nbsp;·&nbsp; z₂ = W₂a₁ + b₂ &nbsp;·&nbsp; L = ½(z₂ − y)²</div>

Work backwards, defining δ = ∂L/∂(pre-activation) at each layer.

**Output layer.** L is an explicit function of z₂:

<div class="eq">δ₂ = ∂L⁄∂z₂ = (z₂ − y)</div>

Since z₂ = W₂a₁ + b₂, and ∂z₂/∂W₂[i,j] = a₁[j]:

<div class="eq">∂L⁄∂W₂ = δ₂a₁ᵀ &nbsp;·&nbsp; ∂L⁄∂b₂ = δ₂</div>

**Hidden layer.** The loss reaches z₁ only through a₁ and then z₂, so the
chain rule gives

<div class="eq">∂L⁄∂a₁ = W₂ᵀδ₂ &nbsp;&nbsp;then&nbsp;&nbsp; δ₁ = ∂L⁄∂z₁ = (W₂ᵀδ₂) ⊙ σ′(z₁)</div>

The ⊙ is elementwise because σ is applied elementwise, so its Jacobian is
diagonal — and multiplying by a diagonal matrix is an elementwise product.
Then, identically to before:

<div class="eq">∂L⁄∂W₁ = δ₁<b>x</b>ᵀ &nbsp;·&nbsp; ∂L⁄∂b₁ = δ₁</div>

**Two patterns repeat at every layer, forever:** propagate backwards by the
*transposed* weight matrix, and multiply elementwise by the activation's
derivative.

### Check the shapes

Shape errors are the most common bug in hand-written backprop, and the
dimensions are forced:

| Quantity | Shape | Why |
|---|---|---|
| **x** | (n,) | input |
| W₁ | (h, n) | maps n → h |
| δ₁ | (h,) | one per hidden pre-activation |
| ∂L/∂W₁ = δ₁**x**ᵀ | (h, n) | must match W₁ |
| W₂ | (1, h) | maps h → 1 |
| δ₂ | (1,) | scalar output |
| W₂ᵀδ₂ | (h,) | back into hidden space |

**A gradient always has the shape of the thing it differentiates.** If yours
does not, stop — you have transposed something.

### Why gradients vanish

δ₁ = (W₂ᵀδ₂) ⊙ σ′(z₁). Across L layers that is a product of L factors of the
form Wᵀ ⊙ σ′. With sigmoid, σ′ = σ(1−σ) ≤ **0.25**, so ten stacked sigmoids
attenuate the gradient by at most 0.25¹⁰ ≈ 10⁻⁶ from the activations alone,
before the weights contribute anything.

That single line is the reason ReLU (σ′ = 1 on the positive half) took over,
and the reason residual connections — which add an identity path whose
derivative is exactly 1 — made 100-layer networks trainable.

```python
"""Backpropagation by hand, checked against numerical gradients."""

import numpy as np


def numeric_jacobian(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Approximates the Jacobian of f at x by central differences.

    Args:
        f: Function from R^n to R^m.
        x: Point of evaluation, shape (n,).
        h: Step size.

    Returns:
        The Jacobian, shape (m, n).
    """
    x = np.asarray(x, dtype=np.float64)
    base = np.atleast_1d(f(x))
    jac = np.zeros((base.size, x.size))
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = h
        jac[:, i] = (np.atleast_1d(f(x + step)) -
                     np.atleast_1d(f(x - step))) / (2.0 * h)
    return jac


def forward_backward(w1, b1, w2, b2, x, y):
    """Runs one forward and backward pass of a two-layer tanh network.

    Args:
        w1: First weight matrix, shape (hidden, in).
        b1: First bias, shape (hidden,).
        w2: Second weight matrix, shape (1, hidden).
        b2: Second bias, shape (1,).
        x: Input vector, shape (in,).
        y: Target scalar.

    Returns:
        A tuple (loss, gradients) where gradients is a dict.
    """
    z1 = w1 @ x + b1
    a1 = np.tanh(z1)
    z2 = w2 @ a1 + b2
    loss = 0.5 * float(((z2 - y) ** 2).item())

    d2 = (z2 - y)                          # dL/dz2
    g_w2 = np.outer(d2, a1)
    g_b2 = d2
    d1 = (w2.T @ d2) * (1.0 - a1 ** 2)     # tanh'(z) = 1 - tanh²(z)
    g_w1 = np.outer(d1, x)
    g_b1 = d1
    return loss, {'w1': g_w1, 'b1': g_b1, 'w2': g_w2, 'b2': g_b2}
```

Check it numerically and the agreement should be about 1e-9. If it is not,
your analytic gradient is wrong — and a wrong gradient will still train, just
badly, which is why this check saves days.

## 7.4 Curvature and convexity

The **Hessian** is the matrix of second partials, and it is symmetric whenever
the second partials are continuous (Clairaut), so Chapter 5's spectral theorem
applies. At a stationary point:

| Hessian | Point |
|---|---|
| positive definite (all λ > 0) | local minimum |
| negative definite (all λ < 0) | local maximum |
| indefinite (mixed signs) | **saddle** |
| singular (some λ = 0) | inconclusive |

A function is **convex** when its Hessian is positive semi-definite
everywhere, and convexity is what makes optimisation honest: every local
minimum is global. Linear and logistic regression are convex; neural networks
are emphatically not.

In high dimensions, though, the usual fear is misplaced. A local minimum
requires *every one* of millions of eigenvalues to be positive simultaneously;
if signs were even roughly independent, that is vanishingly unlikely. Critical
points in large networks are therefore overwhelmingly **saddles**, which
first-order methods escape by following the negative-curvature direction. This
is why gradient descent works far better in practice than one-dimensional
intuition predicts.

## 7.5 The curse of dimensionality

High-dimensional space is not big three-dimensional space, and the arithmetic
is brutal *[MML p.443]*. The volume of a ball of radius r in d dimensions
scales as r^d, so the fraction lying within the outer shell of thickness ε is

<div class="eq">1 − (1 − ε)<sup>d</sup></div>

| d | ε = 0.05 |
|---|---|
| 2 | 9.8% |
| 10 | 40.1% |
| 50 | 92.3% |
| 100 | **99.4%** |

**Essentially all the volume is at the surface.** Two consequences you will
meet constantly:

- **Distances concentrate.** For many distributions the ratio of farthest to
  nearest neighbour tends to 1 as d grows, so "nearest neighbour" degrades
  into "any neighbour", and distance-based methods lose their meaning.
- **Sample requirements explode.** To keep a fixed density you need
  exponentially many points in d.

This is why dimensionality reduction (Chapter 5), regularisation (Chapter 12)
and representation learning (Chapter 20) are not optional refinements. They
are what make learning in high dimensions possible at all.

## 7.6 Gate for this chapter

| Check | Threshold |
|---|---|
| Analytic vs numeric Jacobian on a 3→2 map | < 1e-6 |
| Hand-written backprop vs numerical gradient | < 1e-7 relative |
| Every gradient's shape matches its parameter | exact |
| Shell-volume table reproduced at d = 2, 10, 50, 100 | to 0.1% |
| Rosenbrock with and without momentum | iteration counts recorded for both |

---

# 8 · Probability, information, and where losses come from

## 8.1 The axioms

A probability measure P assigns numbers to events with P(A) ≥ 0, P(Ω) = 1, and
countable additivity for disjoint events *[MML p.498]*. Everything else is
consequence.

**Conditional probability** P(A|B) = P(A∩B)/P(B) *[MML p.515]* rearranges into
Bayes' theorem. The derivation is one line — write P(A∩B) two ways:

<div class="eq">P(A|B)P(B) = P(A∩B) = P(B|A)P(A) &nbsp;⟹&nbsp; P(A|B) = P(B|A)P(A)⁄P(B)</div>

### Worked example — the base rate, in counts

Symbols hide the result; counts reveal it. A test is 99% sensitive and 99%
specific; the disease affects 1 in 1000. Imagine **100,000 people**:

| | Sick (100) | Healthy (99,900) | Total |
|---|---|---|---|
| **Test +** | 99 | 999 | 1,098 |
| **Test −** | 1 | 98,901 | 98,902 |

Of the 1,098 positives, 99 are genuinely sick:

<div class="eq">P(sick | +) = 99⁄1098 ≈ 0.090</div>

About **9%**. A 99%-accurate test on a rare condition produces ten false
positives for every true one, because the healthy group is a thousand times
larger and even a 1% error rate on it swamps the sick group entirely.

Keep this in your pocket: it is the same arithmetic as a fraud detector with a
1% base rate, and it is why precision collapses on imbalanced data no matter
how good the classifier's accuracy looks (Chapter 14).

## 8.2 Random variables and distributions

A random variable maps outcomes to numbers *[MML p.534]*. Discrete ones carry
a probability mass function; continuous ones a density f, where probability is
the integral over a region, and P(X = x) = 0 pointwise *[MML p.567]*.

| Distribution | Use it for | Mean / Variance |
|---|---|---|
| Bernoulli(p) | one binary outcome | p / p(1−p) |
| Binomial(n,p) | successes in n trials | np / np(1−p) |
| Poisson(λ) | counts per interval | λ / λ |
| Normal(μ,σ²) | sums of many small effects | μ / σ² |
| Exponential(λ) | waiting times | 1/λ / 1/λ² |

The Normal earns its ubiquity from the **Central Limit Theorem**: for i.i.d.
Xᵢ with mean μ and finite variance σ²,

<div class="eq">√n(X̄<sub>n</sub> − μ) → Normal(0, σ²) &nbsp;&nbsp;in distribution</div>

regardless of what the Xᵢ individually looked like. That is why measurement
noise is usually modelled as Gaussian — and also why heavy-tailed data
(prices, traffic, incomes), where σ² may not be finite, breaks models that
assume it.

## 8.3 Expectation, variance, and the law of large numbers

<div class="eq">𝔼[X] = Σ<sub>x</sub> x·p(x) &nbsp;&nbsp;or&nbsp;&nbsp; ∫ x·f(x) dx &nbsp;&nbsp;·&nbsp;&nbsp; Var(X) = 𝔼[(X − 𝔼[X])²]</div>

The computational form follows by expanding the square:

<div class="eq">Var(X) = 𝔼[X² − 2X𝔼[X] + 𝔼[X]²] = 𝔼[X²] − 2𝔼[X]² + 𝔼[X]² = 𝔼[X²] − 𝔼[X]²</div>

Expectation is linear *always* — 𝔼[aX + bY] = a𝔼[X] + b𝔼[Y] even for dependent
variables *[MML p.587]*. Variance is not:

<div class="eq">Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)</div>

so it adds only under independence. Nearly every "my confidence interval is
too narrow" bug traces back to that second sentence — correlated errors treated
as independent.

### Why error shrinks like 1/√n

For i.i.d. Xᵢ, linearity gives 𝔼[X̄] = μ, and independence gives

<div class="eq">Var(X̄<sub>n</sub>) = Var((1⁄n)ΣXᵢ) = (1⁄n²)·n·σ² = σ²⁄n</div>

so the standard error is σ/√n. ∎ **Every estimate you report is subject to
that √n**: quadrupling the data halves the error bar, and no amount of
modelling cleverness changes the exponent. It is also why a single train/test
split is noisy, and why Chapter 14 insists on bootstrap intervals.

### Stable variance

Compute variance in one pass with Welford's algorithm rather than the textbook
𝔼[X²] − 𝔼[X]², which catastrophically cancels for large means: with
mean ≈ 10⁹, both terms are ≈ 10¹⁸ while their difference is ≈ 1, and double
precision carries only ~16 digits — so the answer is noise.

```python
"""Welford's online mean and variance."""

import numpy as np


def welford(values: np.ndarray) -> tuple:
    """Computes mean and sample variance in one numerically stable pass.

    Args:
        values: One-dimensional array of observations.

    Returns:
        A tuple (mean, sample_variance).
    """
    count = 0
    mean = 0.0
    m2 = 0.0
    for value in np.asarray(values, dtype=np.float64).ravel():
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
    return mean, m2 / (count - 1) if count > 1 else 0.0


big = np.float64(1e9) + np.random.default_rng(0).normal(size=100_000)
naive = float((big ** 2).mean() - big.mean() ** 2)   # cancels catastrophically
print('naive variance  :', naive)
print('welford variance:', welford(big)[1])
print('numpy variance  :', float(big.var(ddof=1)))
```

## 8.4 Information: entropy, cross-entropy, KL

**Entropy** measures the average surprise in a distribution *[MML p.606]*:

<div class="eq">H(p) = − Σ<sub>x</sub> p(x)·log p(x)</div>

### Worked example — four outcomes

| Distribution | Computation | Entropy (bits) |
|---|---|---|
| (¼,¼,¼,¼) | −4·¼·log₂¼ | **2.000** |
| (½,¼,⅛,⅛) | −(½·−1 + ¼·−2 + 2·⅛·−3) | **1.750** |
| (1,0,0,0) | 0 | **0.000** |

Uniform is maximal (log₂4 = 2 bits), certainty is zero, and the skewed case
sits between. In bits, entropy is literally the optimal average code length —
the second distribution really can be coded in 1.75 bits per symbol on
average, and no scheme does better.

**Cross-entropy** is the cost of coding p with a code built for q:

<div class="eq">H(p, q) = − Σ<sub>x</sub> p(x)·log q(x)</div>

and the excess you pay for being wrong is the **Kullback–Leibler divergence**:

<div class="eq">D<sub>KL</sub>(p ‖ q) = Σ<sub>x</sub> p(x)·log(p(x) ⁄ q(x)) = H(p, q) − H(p)</div>

### Why KL is never negative

This is Gibbs' inequality, and it follows from the concavity of the logarithm
via **Jensen's inequality** (𝔼[φ(X)] ≤ φ(𝔼[X]) for concave φ):

<div class="eq">−D<sub>KL</sub>(p‖q) = Σ<sub>x</sub> p(x)·log(q(x)⁄p(x)) ≤ log Σ<sub>x</sub> p(x)·(q(x)⁄p(x)) = log Σ<sub>x</sub> q(x) = log 1 = 0</div>

so D_KL ≥ 0, with equality exactly when q(x)/p(x) is constant — that is, when
p = q. ∎

KL is also **not symmetric**: D_KL(p‖q) ≠ D_KL(q‖p). Take p = (0.5, 0.5) and
q = (0.9, 0.1):

<div class="eq">D<sub>KL</sub>(p‖q) = 0.5·log(0.5⁄0.9) + 0.5·log(0.5⁄0.1) ≈ 0.511 nats</div>
<div class="eq">D<sub>KL</sub>(q‖p) = 0.9·log(0.9⁄0.5) + 0.1·log(0.1⁄0.5) ≈ 0.368 nats</div>

Different numbers — which is why it is a divergence rather than a distance,
and why the *direction* matters when you fit a VAE (Chapter 20). The forward
direction punishes q for putting zero mass where p has some (mode-covering);
the reverse punishes q for spreading mass where p has none (mode-seeking).

Now the payoff. Since H(p) does not depend on your model, **minimising
cross-entropy is exactly minimising KL divergence to the truth**. The loss
function in every classifier you will ever train is a statement about
information.

## 8.5 Maximum likelihood: where every loss comes from

Given data D and a parametric model, the **likelihood** is P(D|θ), and
maximum likelihood picks the θ making the observed data most probable
*[MML p.619]*. Products underflow, so work with the log — and since log is
monotone, the maximiser is unchanged:

<div class="eq">θ̂ = argmax<sub>θ</sub> Σᵢ log p(xᵢ | θ)</div>

### Derivation 1 — Gaussian noise gives squared error

Assume yᵢ = f(xᵢ;θ) + εᵢ with ε ~ N(0, σ²). The density of one observation is

<div class="eq">p(yᵢ|xᵢ,θ) = (1⁄(σ√(2π)))·exp(−(yᵢ − f(xᵢ;θ))²⁄(2σ²))</div>

Take logs and sum:

<div class="eq">log L = −(1⁄2σ²)·Σᵢ (yᵢ − f(xᵢ;θ))² − n·log σ − (n⁄2)·log(2π)</div>

The last two terms are constant in θ. Maximising log L is therefore
*minimising* Σ(yᵢ − f(xᵢ))² — **least squares**. ∎

Least squares was never an arbitrary choice; it is the MLE under Gaussian
noise. That is also the honest reason to distrust MSE on heavy-tailed targets:
the *assumption* fails, not the arithmetic. Assume Laplace noise instead and
the same derivation yields **MAE**.

### Derivation 2 — Bernoulli outcomes give cross-entropy

With pᵢ = f(xᵢ;θ) ∈ (0,1) and yᵢ ∈ {0,1}, a single observation has probability
pᵢ^{yᵢ}(1−pᵢ)^{1−yᵢ} — which equals pᵢ when yᵢ=1 and 1−pᵢ when yᵢ=0. Logging:

<div class="eq">log L = Σᵢ [ yᵢ·log pᵢ + (1 − yᵢ)·log(1 − pᵢ) ]</div>

Negate and you have exactly the **binary cross-entropy** of *[PTG p.39]*. The
categorical case gives multi-class cross-entropy identically. ∎

**So the two losses that dominate machine learning are not conventions.** They
are maximum-likelihood estimates under two noise models, and choosing a loss
is choosing a probabilistic assumption about your data — whether you realise
it or not.

### Why the MLE variance is biased

Maximising over σ² too gives σ̂² = (1/n)Σ(xᵢ − μ̂)². But

<div class="eq">𝔼[σ̂²] = ((n−1)⁄n)·σ² &lt; σ²</div>

because the same data was used to estimate μ, and the residuals are measured
from the sample mean, which sits closer to the data than the true mean does.
Correcting the factor gives the familiar 1/(n−1). **The MLE is not always
unbiased** — a useful antidote to treating it as automatically correct.

```python
"""Entropy, KL, and maximum likelihood estimates."""

import numpy as np


def entropy(p: np.ndarray, base: float = np.e) -> float:
    """Shannon entropy of a probability vector.

    Args:
        p: Probabilities, summing to 1.
        base: Logarithm base; e gives nats, 2 gives bits.

    Returns:
        The entropy; zero probabilities contribute zero.

    Raises:
        ValueError: If p does not sum to 1 or has negative entries.
    """
    p = np.asarray(p, dtype=np.float64)
    if np.any(p < 0) or abs(p.sum() - 1.0) > 1e-9:
        raise ValueError('p must be a probability vector')
    nonzero = p[p > 0]
    return float(-np.sum(nonzero * np.log(nonzero)) / np.log(base))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence from q to p, in nats.

    Args:
        p: Reference distribution.
        q: Comparison distribution.

    Returns:
        The divergence, always >= 0.

    Raises:
        ValueError: If q is zero where p is positive.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    if np.any(q[mask] <= 0):
        raise ValueError('KL is infinite where q = 0 but p > 0')
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def mle_gaussian(samples: np.ndarray) -> tuple:
    """Maximum-likelihood mean and variance of a Gaussian.

    Note the 1/n divisor: the MLE of the variance is biased, which is
    why the unbiased estimator uses 1/(n-1) instead.

    Args:
        samples: Observed values.

    Returns:
        A tuple (mu_hat, sigma_squared_hat).
    """
    x = np.asarray(samples, dtype=np.float64)
    mu = float(x.mean())
    return mu, float(((x - mu) ** 2).mean())
```

## 8.6 Gate for this chapter

| Check | Threshold |
|---|---|
| Entropy of uniform over n | log n, to 1e-12 |
| Your hand-computed entropies (2.000, 1.750, 0.000 bits) | exact |
| D_KL(p‖p) = 0 and D_KL ≥ 0 over 1000 random pairs | exact / non-negative |
| Asymmetry example (0.511 vs 0.368 nats) | to 1e-3 |
| MLE recovers μ, σ² from 1e5 samples | within 3 standard errors |
| Welford vs naive variance at mean 1e9 | naive visibly wrong |
| Bayes worked example | 0.090 ± 0.001 |
