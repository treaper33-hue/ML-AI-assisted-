# Part I — The Mathematical Engine

Machine learning is three pieces of mathematics wearing a trench coat: linear
algebra moves the data, calculus moves the parameters, probability tells you
what any of it means. Part I builds all three from the ground, in code.

The rule for this part is **implement before you use**. Everything here exists
in NumPy. Writing it once yourself is what converts a formula you have seen
into a tool you own — and NumPy then serves as the oracle that tells you
whether you got it right.

---

# 1 · Vectors, spaces, and the array

## 1.1 What a vector space actually buys you

A **vector space** over ℝ is a set V with addition and scalar multiplication
that behave the way you already expect: addition is commutative and
associative, there is a zero, every element has a negative, and scalar
multiplication distributes over both kinds of addition *[MML p.9]*.

That list of axioms looks like bookkeeping. It is not. It is a *licence*. Once
you know a set is a vector space, every theorem about vector spaces applies to
it — and the set does not have to look like arrows. These are all vector
spaces:

- ℝⁿ, the n-tuples of real numbers — a row of a dataset;
- the real-valued functions on an interval — a signal, a density;
- the polynomials of degree ≤ k — a feature expansion;
- the m×n matrices — images, weight layers.

This is why a single body of theory covers tabular rows, images and
distributions at once. When you later read that "the activations live in a
512-dimensional space," that sentence is doing real work: it means every tool
in this part applies to them.

## 1.2 Span, independence, basis

A **linear combination** of **v**₁, …, **v**ₖ is any c₁**v**₁ + ⋯ + cₖ**v**ₖ.
The set of all of them is the **span** *[MML p.16]*.

The vectors are **linearly independent** when the only combination equal to
zero is the one with every cᵢ = 0 *[MML p.15]*. Equivalently — and this is the
form you will use — none of them is reachable from the others, so none is
redundant.

A **basis** is a linearly independent set that spans the whole space: the
minimal description with nothing left over *[MML p.18]*. The number of basis
vectors is the **dimension**.

### Worked example — uniqueness of coordinates

Why does independence matter? Because it is exactly what makes coordinates
*unique*, and uniqueness is what lets you speak of "the" representation of a
data point at all.

Suppose a vector **x** had two representations in the same basis:

<div class="eq"><b>x</b> = Σᵢ aᵢ<b>v</b>ᵢ = Σᵢ bᵢ<b>v</b>ᵢ</div>

Subtract them: Σᵢ(aᵢ − bᵢ)**v**ᵢ = **0**. By independence every coefficient of
that combination must be zero, so aᵢ = bᵢ for all i. The two representations
were the same one. ∎

Lose independence and the proof collapses — and with it, the meaning of a
coefficient. Take **v**₁ = (1,0), **v**₂ = (0,1), **v**₃ = (1,1), and the
vector **x** = (2,3):

<div class="eq"><b>x</b> = 2<b>v</b>₁ + 3<b>v</b>₂ + 0<b>v</b>₃ = 1<b>v</b>₁ + 2<b>v</b>₂ + 1<b>v</b>₃ = 0<b>v</b>₁ + 1<b>v</b>₂ + 2<b>v</b>₃</div>

Three different "explanations" of the same point, all equally valid. **This is
collinearity in a regression, stated exactly.** The predictions are identical;
the coefficients are arbitrary; and any story you tell about "the effect of
feature 3" is a story about which of infinitely many solutions your solver
happened to land on.

> **Practitioner's rule.** `np.linalg.matrix_rank(X) < X.shape[1]` means
> redundancy, and it means your coefficients are not identifiable. Ridge
> (Chapter 12) does not merely shrink them — it *picks one*, by adding a
> criterion the data could not supply.

## 1.3 Data as a matrix

Stack n observations of d features and you have the **design matrix**
X ∈ ℝ^(n×d) *[MML p.110]*. Two readings of the same object, both useful:

- **by rows** — n points in a d-dimensional feature space (the geometric view:
  clustering, distances, neighbourhoods);
- **by columns** — d vectors in an n-dimensional sample space (the algebraic
  view: collinearity, rank, least squares).

Most confusion about a model evaporates when you decide, deliberately, which
of these two pictures you are in.

## 1.4 The array is not a list

NumPy's `ndarray` is a typed, fixed-size, contiguous block with a shape.
Python's `list` is a box of pointers to arbitrary objects. That difference is
the whole performance story *[MML p.29, p.35]*.

A Python loop over a list pays, per element, for pointer chasing, type
dispatch and the construction of a new boxed float. NumPy pays once for the
whole array and runs a tight typed loop in C. The asymptotics are identical —
the constant is one to two orders of magnitude *[MML p.37]*.

```python
"""Measures the cost of a Python loop against a vectorised dot product."""

import time

import numpy as np


def dot_python(a: list, b: list) -> float:
    """Computes a dot product with a Python loop."""
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def benchmark(n: int) -> tuple:
    """Times both implementations at length n.

    Args:
        n: Vector length.

    Returns:
        A tuple of (python_seconds, numpy_seconds, speedup).
    """
    rng = np.random.default_rng(0)
    arr_a, arr_b = rng.normal(size=n), rng.normal(size=n)
    list_a, list_b = arr_a.tolist(), arr_b.tolist()

    start = time.perf_counter()
    slow = dot_python(list_a, list_b)
    python_s = time.perf_counter() - start

    start = time.perf_counter()
    fast = float(np.dot(arr_a, arr_b))
    numpy_s = time.perf_counter() - start

    assert abs(slow - fast) < 1e-8 * max(1.0, abs(fast))
    return python_s, numpy_s, python_s / max(numpy_s, 1e-12)


for size in (10_000, 100_000, 1_000_000):
    py_s, np_s, ratio = benchmark(size)
    print(f'n={size:>9,}  python {py_s:7.4f}s  numpy {np_s:7.5f}s  '
          f'speedup {ratio:6.0f}x')
```

Run it. Do not take the number on faith and do not take mine — the ratio
depends on your BLAS, your cache and your CPU, and no source in this stack
publishes a figure you could check against. Record what *your* machine does.

**Broadcasting** is the other half of thinking in arrays. When shapes differ,
NumPy compares them from the right and stretches any axis of length 1:

```python
x = np.arange(12).reshape(4, 3)       # (4, 3)
column_means = x.mean(axis=0)         # (3,)
centred = x - column_means            # (4,3) - (3,) -> (4,3)
```

That last line is feature centring, the first step of PCA, written without a
loop. The mental discipline: *never write a loop over samples.*

### A worked trap: the silent broadcast

Broadcasting is also the most common silent bug in array code. Consider
computing residuals where `y` has shape (n,) and `y_pred` has shape (n, 1):

<div class="eq">(n,) − (n,1) → (n,n)</div>

NumPy aligns from the right, sees (n,) as (1,n), and produces an n×n matrix of
*every pairwise difference*. Your RMSE then averages n² numbers, most of them
meaningless, and returns a plausible-looking float. Nothing raises.

```python
y = np.array([1.0, 2.0, 3.0])
y_pred = np.array([[1.1], [1.9], [3.2]])
print((y - y_pred).shape)             # (3, 3)  -- not (3,)
print(float(np.sqrt(np.mean((y - y_pred) ** 2))))   # 0.99..., nonsense
print(float(np.sqrt(np.mean((y - y_pred.ravel()) ** 2))))   # 0.14, correct
```

This is why every metric in this text validates shapes and refuses mismatches
rather than broadcasting them.

## 1.5 Gate for this chapter

| Check | Threshold |
|---|---|
| `matrix_rank` detects a deliberately collinear column | rank = d − 1 |
| Your vectorised dot matches the loop | within 1e-8 relative |
| You can state the speedup on your machine | a number, recorded |
| You can reproduce the three-representation example | exact |

---

# 2 · Geometry: norms, angles, orthogonality

## 2.1 Norms

A **norm** assigns a length ‖**x**‖ with three properties: it is zero only for
the zero vector, it scales (‖c**x**‖ = |c|‖**x**‖), and it satisfies the
triangle inequality *[MML p.49]*. The family you will meet:

<div class="eq">‖<b>x</b>‖₁ = Σᵢ |xᵢ| &nbsp;&nbsp;·&nbsp;&nbsp; ‖<b>x</b>‖₂ = √(Σᵢ xᵢ²) &nbsp;&nbsp;·&nbsp;&nbsp; ‖<b>x</b>‖∞ = maxᵢ |xᵢ|</div>

### Worked example — the same vector, three lengths

Take **x** = (3, −4, 12):

| Norm | Computation | Value |
|---|---|---|
| ‖**x**‖₁ | 3 + 4 + 12 | **19** |
| ‖**x**‖₂ | √(9 + 16 + 144) = √169 | **13** |
| ‖**x**‖∞ | max(3, 4, 12) | **12** |

Three honest answers to "how big is this?", and they are ordered
‖x‖∞ ≤ ‖x‖₂ ≤ ‖x‖₁ — which holds in general and is worth knowing, because it
tells you an L1 penalty is always at least as aggressive as an L2 one at the
same coefficient size.

The unit balls differ in a way that has consequences you will meet in
Chapter 12. The L2 ball is round. The L1 ball is a diamond with **corners on
the axes** — and a corner on an axis is a point where some coordinates are
exactly zero. That geometric fact is the whole explanation for L1's
sparsity, and we will finish the argument when we have the loss contours.

Every norm induces a **distance**, d(**x**, **y**) = ‖**x** − **y**‖
*[MML p.55]*, which is where k-nearest-neighbours, k-means and every
clustering algorithm get their notion of "near".

## 2.2 Inner products and angle

The **inner product** ⟨**x**, **y**⟩ = Σᵢ xᵢyᵢ = **x**ᵀ**y** generates the
Euclidean norm, ‖**x**‖₂ = √⟨**x**, **x**⟩ *[MML p.58]*, and with it the
angle between vectors *[MML p.61]*:

<div class="eq">cos θ = ⟨<b>x</b>, <b>y</b>⟩ ⁄ (‖<b>x</b>‖₂ ‖<b>y</b>‖₂)</div>

### Why that formula is the cosine

This is usually asserted. It follows from the law of cosines in one step, and
the derivation is worth seeing because it explains why dot products measure
*alignment*.

The law of cosines on the triangle with sides **x**, **y** and **x** − **y**:

<div class="eq">‖<b>x</b> − <b>y</b>‖² = ‖<b>x</b>‖² + ‖<b>y</b>‖² − 2‖<b>x</b>‖‖<b>y</b>‖cos θ</div>

Expand the left side algebraically instead:

<div class="eq">‖<b>x</b> − <b>y</b>‖² = ⟨<b>x</b> − <b>y</b>, <b>x</b> − <b>y</b>⟩ = ‖<b>x</b>‖² − 2⟨<b>x</b>, <b>y</b>⟩ + ‖<b>y</b>‖²</div>

Set the two right-hand sides equal; ‖**x**‖² and ‖**y**‖² cancel, leaving
−2⟨**x**,**y**⟩ = −2‖**x**‖‖**y**‖cos θ, hence the formula. ∎

So **cosine similarity is not a metaphor** — it is the actual cosine of the
actual angle, and that is why it discards magnitude and keeps direction, which
is exactly right when a long document and a short one should count as similar
if they are about the same thing.

**Orthogonality** is ⟨**x**, **y**⟩ = 0 *[MML p.60]*: zero correlation in the
geometric sense, no shared component, nothing of one inside the other.

## 2.3 Projection, derived

The projection of **x** onto the direction of **v** is the piece of **x**
lying along **v**. Rather than quote it, derive it: we want the scalar c that
makes the residual **x** − c**v** orthogonal to **v**.

<div class="eq">⟨<b>x</b> − c<b>v</b>, <b>v</b>⟩ = 0 &nbsp;⟹&nbsp; ⟨<b>x</b>, <b>v</b>⟩ = c⟨<b>v</b>, <b>v</b>⟩ &nbsp;⟹&nbsp; c = ⟨<b>x</b>, <b>v</b>⟩ ⁄ ⟨<b>v</b>, <b>v</b>⟩</div>

<div class="eq">proj<sub><b>v</b></sub>(<b>x</b>) = (⟨<b>x</b>, <b>v</b>⟩ ⁄ ⟨<b>v</b>, <b>v</b>⟩) <b>v</b></div>

That one-line argument — *choose the coefficient that makes the residual
orthogonal* — reappears as the normal equations in Chapter 4, as PCA in
Chapter 5, and as the conditional expectation in Chapter 8. It is the single
most reused idea in this book.

### Worked example — projection by hand

Project **x** = (3, 4) onto **v** = (1, 0):

<div class="eq">c = (3·1 + 4·0) ⁄ (1·1 + 0·0) = 3 &nbsp;⟹&nbsp; proj = (3, 0), &nbsp; residual = (0, 4)</div>

Residual ⟂ **v**, as designed. Now the matrix form. The projection *matrix*
onto span(**v**) is

<div class="eq">P = <b>vv</b>ᵀ ⁄ ⟨<b>v</b>, <b>v</b>⟩</div>

and it has two properties worth checking by hand. **Idempotent**: projecting
twice is projecting once, P² = P, because

<div class="eq">P² = (<b>vv</b>ᵀ<b>vv</b>ᵀ) ⁄ ⟨<b>v</b>,<b>v</b>⟩² = <b>v</b>(<b>v</b>ᵀ<b>v</b>)<b>v</b>ᵀ ⁄ ⟨<b>v</b>,<b>v</b>⟩² = <b>vv</b>ᵀ ⁄ ⟨<b>v</b>,<b>v</b>⟩ = P</div>

And **symmetric**: P = Pᵀ, since (**vv**ᵀ)ᵀ = **vv**ᵀ. Those two properties
*characterise* orthogonal projections — any symmetric idempotent matrix is one.

When the basis is **orthonormal**, coordinates become free: the i-th
coordinate of **x** is simply ⟨**x**, **u**ᵢ⟩, with no system to solve
*[MML p.65]*. This is why orthogonal bases are hunted so persistently — they
turn a linear solve into a dot product.

## 2.4 Gram–Schmidt

Given independent **v**₁, …, **v**ₖ, build an orthonormal basis for the same
span *[MML p.67, code p.89]*. Take each vector, subtract its projections onto
everything already fixed, normalise what remains.

### Worked example — three vectors, by hand

Start with **v**₁ = (1, 1, 0), **v**₂ = (1, 0, 1), **v**₃ = (0, 1, 1).

**Step 1.** **u**₁ = **v**₁/‖**v**₁‖ = (1, 1, 0)/√2.

**Step 2.** Remove from **v**₂ its component along **u**₁:

<div class="eq">⟨<b>v</b>₂, <b>u</b>₁⟩ = (1 + 0 + 0)⁄√2 = 1⁄√2</div>
<div class="eq"><b>w</b>₂ = (1, 0, 1) − (1⁄√2)·(1, 1, 0)⁄√2 = (1, 0, 1) − (½, ½, 0) = (½, −½, 1)</div>

‖**w**₂‖ = √(¼ + ¼ + 1) = √(3/2), so **u**₂ = (½, −½, 1)/√(3/2).

**Step 3.** Remove from **v**₃ its components along both:

<div class="eq">⟨<b>v</b>₃, <b>u</b>₁⟩ = 1⁄√2, &nbsp;&nbsp; ⟨<b>v</b>₃, <b>u</b>₂⟩ = (0 − ½ + 1)⁄√(3/2) = ½⁄√(3/2)</div>
<div class="eq"><b>w</b>₃ = (0,1,1) − (½,½,0) − (⅙,−⅙,⅓) = (−⅔, ⅔, ⅔)</div>

Check: ⟨**w**₃, **u**₁⟩ = (−⅔ + ⅔)/√2 = 0 ✓, and ⟨**w**₃, **u**₂⟩ =
(−⅓ − ⅓ + ⅔)/√(3/2) = 0 ✓. Normalise and you have an orthonormal basis.

Do that once with a pencil. The code will never be mysterious again.

```python
"""Classical and modified Gram-Schmidt, and the difference between them."""

import numpy as np


def gram_schmidt_classical(a: np.ndarray) -> np.ndarray:
    """Orthonormalises columns by subtracting all projections at once.

    Args:
        a: Matrix of shape (n, k) with linearly independent columns.

    Returns:
        Matrix Q of shape (n, k) with orthonormal columns.

    Raises:
        ValueError: If the columns are not linearly independent.
    """
    n, k = a.shape
    q = np.zeros((n, k), dtype=np.float64)
    for j in range(k):
        v = a[:, j].astype(np.float64).copy()
        # Project onto the already-fixed directions, using the ORIGINAL
        # column each time -- this is what makes it "classical".
        for i in range(j):
            v -= (q[:, i] @ a[:, j]) * q[:, i]
        norm = np.linalg.norm(v)
        if norm < 1e-13:
            raise ValueError(f'column {j} is dependent on the earlier ones')
        q[:, j] = v / norm
    return q


def gram_schmidt_modified(a: np.ndarray) -> np.ndarray:
    """Orthonormalises columns, re-projecting against the running residual.

    Numerically this is the one to use: each projection is computed from
    the partially orthogonalised vector, so rounding error is not allowed
    to accumulate unchecked.

    Args:
        a: Matrix of shape (n, k) with linearly independent columns.

    Returns:
        Matrix Q of shape (n, k) with orthonormal columns.

    Raises:
        ValueError: If the columns are not linearly independent.
    """
    n, k = a.shape
    q = np.zeros((n, k), dtype=np.float64)
    for j in range(k):
        v = a[:, j].astype(np.float64).copy()
        for i in range(j):
            v -= (q[:, i] @ v) * q[:, i]      # note: v, not a[:, j]
        norm = np.linalg.norm(v)
        if norm < 1e-13:
            raise ValueError(f'column {j} is dependent on the earlier ones')
        q[:, j] = v / norm
    return q


def orthogonality_error(q: np.ndarray) -> float:
    """Returns the largest deviation of QᵀQ from the identity."""
    return float(np.abs(q.T @ q - np.eye(q.shape[1])).max())
```

### Why the two versions differ

Mathematically they are identical: in exact arithmetic
⟨**u**ᵢ, **v**ⱼ⟩ = ⟨**u**ᵢ, **w**⟩ for the partially orthogonalised **w**,
because the pieces already removed were orthogonal to **u**ᵢ.

In floating point they are not. The classical version computes every
projection coefficient from the *original* **v**ⱼ, so an error made at step 1
is never seen again — it propagates untouched into every later step. The
modified version recomputes against the running residual, so each step sees,
and partially corrects, the error so far.

A **Hilbert matrix**, hᵢⱼ = 1/(i+j+1), is famously ill-conditioned; run both
on one:

```python
n = 10
hilbert = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
print('condition number     :', f'{np.linalg.cond(hilbert):.3e}')
print('classical error      :', orthogonality_error(gram_schmidt_classical(hilbert)))
print('modified error       :', orthogonality_error(gram_schmidt_modified(hilbert)))
print('numpy qr error       :', orthogonality_error(np.linalg.qr(hilbert)[0]))
```

The classical version loses orthogonality catastrophically; the modified one
survives; LAPACK's QR (Householder reflections, a different algorithm
entirely) is better still. **This is the lesson of Part I in a single
experiment:** correct mathematics and correct numerics are two different
achievements, and the gap between them widens exactly where your data is most
structured.

## 2.5 Gate for this chapter

| Check | Threshold |
|---|---|
| ‖(3,4)‖₂ | exactly 5.0 |
| Your hand-worked Gram–Schmidt matches the code | to 1e-12 |
| QᵀQ = I for a well-conditioned input | max deviation < 1e-10 |
| Classical vs modified on a 10×10 Hilbert matrix | classical error > modified by orders of magnitude |
| Projection is idempotent and symmetric | P² = P, P = Pᵀ to 1e-12 |

---

# 3 · Matrices as linear maps

## 3.1 The map is the point

A matrix is not a grid of numbers. It is a **linear map** written down in a
particular basis *[MML p.117]*. A function T is linear when

<div class="eq">T(a<b>x</b> + b<b>y</b>) = a·T(<b>x</b>) + b·T(<b>y</b>)</div>

and every linear map between finite-dimensional spaces *is* a matrix, once you
fix bases *[MML p.121]*.

### Why composition is matrix multiplication

The definition of matrix multiplication looks arbitrary until you derive it
from composition, which takes three lines. Let B send basis vector **e**ⱼ to
the column Σₖ bₖⱼ**e**ₖ, and let A send **e**ₖ to Σᵢ aᵢₖ**e**ᵢ. Then

<div class="eq">(A∘B)(<b>e</b><sub>j</sub>) = A(Σ<sub>k</sub> b<sub>kj</sub><b>e</b><sub>k</sub>) = Σ<sub>k</sub> b<sub>kj</sub>·A(<b>e</b><sub>k</sub>) = Σ<sub>k</sub> b<sub>kj</sub> Σᵢ a<sub>ik</sub><b>e</b>ᵢ = Σᵢ (Σ<sub>k</sub> a<sub>ik</sub>b<sub>kj</sub>) <b>e</b>ᵢ</div>

The coefficient of **e**ᵢ is Σₖ aᵢₖbₖⱼ — precisely the (i,j) entry of AB. ∎

So the "row times column" rule is not a convention; it is what composition
*forces*. Three consequences worth stating plainly:

- **AB means "do B, then A".** Non-commutativity is not a quirk: doing things
  in the other order is a different thing.
- **The columns are the images of the basis vectors.** A**e**ⱼ is the j-th
  column. To know what a matrix *does*, look at where it sends the axes.
- **A neural network layer is a linear map plus a shift plus a
  nonlinearity.** Strip the nonlinearity and the deepest network collapses
  into one matrix — which is exactly why the nonlinearity is not optional.

## 3.2 The four quantities

| Quantity | Definition | What it tells you |
|---|---|---|
| **Image** (column space) | {A**x** : **x** ∈ ℝⁿ} | what outputs are reachable |
| **Kernel** (null space) | {**x** : A**x** = **0**} | what inputs are destroyed |
| **Rank** | dim of the image | how much information survives |
| **Nullity** | dim of the kernel | how much is lost |

tied together by **rank + nullity = n** *[MML p.127]*. The proof is a basis
count: take a basis of the kernel, extend it to a basis of the whole space,
and show the images of the added vectors form a basis of the image. The
counting then gives the identity directly.

A nonzero kernel means the map is not invertible: distinct inputs collide, and
no algorithm can undo that. When a linear model has a nonzero kernel in its
design matrix, there is an entire subspace of coefficient vectors producing
identical predictions — Section 1.2's three representations, in general form.

## 3.3 Transformations you can see

In the plane, every 2×2 matrix is some combination of five moves
*[MML p.133–138]*:

| Move | Matrix | Determinant |
|---|---|---|
| Stretch | [[a,0],[0,b]] | ab |
| Rotation by θ | [[cos θ, −sin θ],[sin θ, cos θ]] | 1 |
| Shear | [[1,k],[0,1]] | 1 |
| Reflection | [[1,0],[0,−1]] | −1 |
| Projection | [[1,0],[0,0]] | 0 |

Read the determinant column. Rotation and shear preserve area; reflection
preserves area but flips orientation; projection destroys area entirely — and
a determinant of zero is precisely the signature of information that cannot be
recovered.

### Worked example — the shear preserves area

Take the unit square with corners (0,0), (1,0), (1,1), (0,1) and apply the
shear [[1,2],[0,1]]. The corners map to (0,0), (1,0), (3,1), (2,1): a slanted
parallelogram. Its area is base × height = 1 × 1 = 1, unchanged, matching
det = 1·1 − 2·0 = 1.

The shape is unrecognisable; the area is identical. That is what "determinant
measures volume, not form" means in practice.

## 3.4 The determinant

The determinant is the **signed volume scaling factor** of the map
*[MML p.139–140]*. Everything else about it follows from that one sentence:

- det(AB) = det(A)det(B) — scalings compose by multiplying;
- det(A) = 0 ⟺ A is singular — the map flattens space, so it cannot be undone;
- |det(A)| is the factor by which any region's volume changes;
- a negative sign means orientation was flipped;
- det(A⁻¹) = 1/det(A), since undoing a scaling scales by the reciprocal.

In practice you never compute it by cofactor expansion — that is O(n!) and
unusable past n ≈ 12 *[MML p.180]*. You factor the matrix first and multiply
the diagonal, which is Chapter 4.

## 3.5 Gate for this chapter

| Check | Threshold |
|---|---|
| Your 2×2 gallery reproduces the determinant column above | exact |
| rank + nullity = n on random rank-deficient matrices | exact, 100 trials |
| Rotation preserves ‖**x**‖₂ | within 1e-12 |
| The sheared unit square has area 1 | exact |

---

# 4 · Solving linear systems

## 4.1 Elimination

Solving A**x** = **b** by hand means **Gaussian elimination**: use row
operations to reach an upper-triangular system, then back-substitute
*[MML p.160]*.

### Worked example — elimination with numbers

Solve

<div class="eq">2x + y − z = 8 &nbsp;·&nbsp; −3x − y + 2z = −11 &nbsp;·&nbsp; −2x + y + 2z = −3</div>

Eliminate x using row 1 (pivot 2). Row 2 ← Row 2 + 1.5·Row 1; Row 3 ←
Row 3 + Row 1:

<div class="eq">2x + y − z = 8 &nbsp;·&nbsp; 0.5y + 0.5z = 1 &nbsp;·&nbsp; 2y + z = 5</div>

Eliminate y using the new row 2 (pivot 0.5). Row 3 ← Row 3 − 4·Row 2:

<div class="eq">2x + y − z = 8 &nbsp;·&nbsp; 0.5y + 0.5z = 1 &nbsp;·&nbsp; −z = 1</div>

Back-substitute: z = −1, then 0.5y − 0.5 = 1 so y = 3, then 2x + 3 + 1 = 8 so
x = 2. **(x, y, z) = (2, 3, −1)**, which you can verify in the original
equations in ten seconds.

Count the work: roughly n³/3 multiply-adds, i.e. **O(n³)** *[MML p.165]*. That
single exponent explains a surprising amount of practical machine learning —
doubling the feature count of an exact linear solve costs eight times the
work.

### Why pivoting is not optional

Consider, in three-digit arithmetic:

<div class="eq">0.001x + y = 1 &nbsp;·&nbsp; x + y = 2</div>

Without pivoting, the multiplier is 1/0.001 = 1000, and row 2 becomes
(1 − 1000)y = 2 − 1000, i.e. −999y = −998, giving y = 0.999 and then
x = (1 − 0.999)/0.001 = 1. Rounded to three digits, y = 1.00 and
x = (1 − 1.00)/0.001 = **0** — catastrophically wrong.

Swap the rows first (partial pivoting puts the largest available pivot on the
diagonal) and the multiplier is 0.001, the arithmetic is benign, and you get
x ≈ 1.00, y ≈ 1.00. Same mathematics, different numerics, completely different
answer.

## 4.2 LU

Elimination can be *recorded* rather than merely performed. The record is the
**LU decomposition** *[MML p.168, code p.171]*:

<div class="eq">PA = LU</div>

P a permutation (the row swaps), L unit lower-triangular (the multipliers), U
upper-triangular (the result). Having it, each new right-hand side costs only
O(n²): solve L**y** = P**b** forwards, then U**x** = **y** backwards. This is
the standard pattern — **factor once, solve many times** — and it is why
`scipy.linalg.lu_factor` exists separately from `lu_solve`.

For the worked example above, the multipliers were −1.5 and −1 (first column)
and 4 (second), so

<div class="eq">L = [[1,0,0],[−1.5,1,0],[−1,4,1]], &nbsp; U = [[2,1,−1],[0,0.5,0.5],[0,0,−1]]</div>

and LU reproduces A exactly. Multiply them out once by hand; it is the
clearest possible demonstration that L and U are just elimination's
bookkeeping.

```python
"""LU decomposition with partial pivoting, and solving with it."""

import numpy as np


def lu_decompose(a: np.ndarray) -> tuple:
    """Factors a square matrix as PA = LU with partial pivoting.

    Args:
        a: Square matrix of shape (n, n).

    Returns:
        A tuple (P, L, U) of ndarrays.

    Raises:
        ValueError: If `a` is not square or is singular.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f'expected a square matrix, got {a.shape}')
    n = a.shape[0]
    u = a.astype(np.float64).copy()
    l = np.eye(n)
    p = np.eye(n)
    for k in range(n - 1):
        pivot = int(np.argmax(np.abs(u[k:, k]))) + k   # partial pivoting
        if abs(u[pivot, k]) < 1e-14:
            raise ValueError('matrix is singular to working precision')
        if pivot != k:
            u[[k, pivot], k:] = u[[pivot, k], k:]
            l[[k, pivot], :k] = l[[pivot, k], :k]
            p[[k, pivot], :] = p[[pivot, k], :]
        for row in range(k + 1, n):
            factor = u[row, k] / u[k, k]
            l[row, k] = factor
            u[row, k:] -= factor * u[k, k:]
    return p, l, u


def lu_solve(p: np.ndarray, l: np.ndarray, u: np.ndarray,
             b: np.ndarray) -> np.ndarray:
    """Solves Ax = b given a PA = LU factorisation.

    Args:
        p: Permutation matrix.
        l: Unit lower-triangular factor.
        u: Upper-triangular factor.
        b: Right-hand side of shape (n,).

    Returns:
        The solution vector x.
    """
    pb = p @ b
    n = len(b)
    y = np.zeros(n)
    for i in range(n):                      # forward substitution
        y[i] = pb[i] - l[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):          # back substitution
        x[i] = (y[i] - u[i, i + 1:] @ x[i + 1:]) / u[i, i]
    return x
```

The determinant now costs nothing extra: det(A) = (−1)^swaps · Π uᵢᵢ
*[MML p.182]*. In the example, det = (−1)⁰ · 2 · 0.5 · (−1) = **−1**.

## 4.3 Never invert

The single most common numerical error in applied work is computing A⁻¹ and
multiplying *[MML p.177]*. Inversion costs about three times a solve, and it
loses accuracy: it computes n separate solves and then makes you push the
result through a matrix multiply, compounding each error.

```python
rng = np.random.default_rng(0)
n = 200
base = rng.normal(size=(n, n))
# Bend the spectrum to make the system badly conditioned on purpose.
u_mat, _, vt = np.linalg.svd(base)
spectrum = np.logspace(0, -10, n)
a = (u_mat * spectrum) @ vt
x_true = rng.normal(size=n)
b = a @ x_true

x_inv = np.linalg.inv(a) @ b
x_solve = np.linalg.solve(a, b)
print(f'condition number : {np.linalg.cond(a):.3e}')
print(f'error via inv    : {np.linalg.norm(x_inv - x_true):.3e}')
print(f'error via solve  : {np.linalg.norm(x_solve - x_true):.3e}')
```

### The condition number, derived

Why does conditioning amplify error? Perturb the right-hand side by δ**b**.
The solution moves by δ**x** = A⁻¹δ**b**, so

<div class="eq">‖δ<b>x</b>‖ ≤ ‖A⁻¹‖·‖δ<b>b</b>‖ &nbsp;&nbsp;and&nbsp;&nbsp; ‖<b>b</b>‖ = ‖A<b>x</b>‖ ≤ ‖A‖·‖<b>x</b>‖</div>

Divide the first by ‖**x**‖ and use the second:

<div class="eq">‖δ<b>x</b>‖⁄‖<b>x</b>‖ ≤ ‖A‖·‖A⁻¹‖ · ‖δ<b>b</b>‖⁄‖<b>b</b>‖ = κ(A)·‖δ<b>b</b>‖⁄‖<b>b</b>‖</div>

So **κ(A) = ‖A‖‖A⁻¹‖ = σ_max/σ_min is exactly the worst-case amplification
factor** from relative input error to relative output error. ∎

A useful rule of thumb follows: at κ ≈ 10ᵏ you lose about k significant digits.
Double precision carries ~16, so κ ≈ 10¹⁶ leaves you none. When a linear
model's coefficients swing wildly for an invisible change in the data, this is
the mechanism — and the fix is regularisation (Chapter 12), which raises
σ_min off the floor.

## 4.4 Least squares, derived

An over-determined system — n observations, d < n unknowns — has no exact
solution, so you ask for the **closest** one: minimise ‖A**x** − **b**‖₂.
Two derivations, and it is worth seeing both.

**Geometrically.** The achievable vectors A**x** form the column space.
The closest point in a subspace is the orthogonal projection (Section 2.3), so
the residual must be orthogonal to *every* column of A:

<div class="eq">Aᵀ(<b>b</b> − A<b>x</b>) = <b>0</b> &nbsp;⟹&nbsp; AᵀA<b>x</b> = Aᵀ<b>b</b></div>

**By calculus.** Expand the objective and differentiate:

<div class="eq">f(<b>x</b>) = ‖A<b>x</b> − <b>b</b>‖² = <b>x</b>ᵀAᵀA<b>x</b> − 2<b>b</b>ᵀA<b>x</b> + <b>b</b>ᵀ<b>b</b></div>
<div class="eq">∇f = 2AᵀA<b>x</b> − 2Aᵀ<b>b</b> = <b>0</b> &nbsp;⟹&nbsp; AᵀA<b>x</b> = Aᵀ<b>b</b></div>

The same **normal equations**, reached from geometry and from calculus. This
is linear regression, and you will meet it again as a maximum-likelihood
estimate in Chapter 8, as ridge regression in Chapter 12, and as the first
thing a neural network learns to imitate in Chapter 18.

One warning that costs real accuracy: κ(AᵀA) = κ(A)². Forming the normal
equations *squares* your conditioning problem. For well-behaved data it is
fine and fast; for anything delicate, solve via QR or SVD instead, which is
what `np.linalg.lstsq` does for you.

## 4.5 Gate for this chapter

| Check | Threshold |
|---|---|
| Your hand-solved 3×3 system | (2, 3, −1) exactly |
| ‖PA − LU‖ | < 1e-10 |
| Your `lu_solve` against `np.linalg.solve` | agree to 1e-10 |
| det via LU against `np.linalg.det` | agree to 1e-10 relative |
| `inv`-and-multiply vs `solve` on κ ≈ 10¹⁰ | solve is measurably more accurate |
| Elimination timing at n = 64, 128, 256 | fitted exponent ≈ 3 |

---

# 5 · Eigenvalues, the spectral theorem, and the SVD

## 5.1 Eigenvectors

**v** ≠ **0** is an **eigenvector** of A with **eigenvalue** λ when

<div class="eq">A<b>v</b> = λ<b>v</b></div>

*[MML p.189]*: a direction the map does not rotate, only scales. Find them in
principle through the characteristic polynomial det(A − λI) = 0
*[MML p.192]* — and never in practice past tiny matrices, because polynomial
root-finding is catastrophically unstable. Real algorithms iterate instead
*[MML p.217, p.230]*.

### Worked example — a 2×2 by hand

Take A = [[4, 1], [2, 3]]. The characteristic polynomial:

<div class="eq">det(A − λI) = (4 − λ)(3 − λ) − 2 = λ² − 7λ + 10 = (λ − 5)(λ − 2)</div>

So λ₁ = 5, λ₂ = 2. For λ = 5, solve (A − 5I)**v** = **0**:

<div class="eq">[[−1, 1], [2, −2]]<b>v</b> = <b>0</b> &nbsp;⟹&nbsp; v₁ = v₂ &nbsp;⟹&nbsp; <b>v</b>₁ ∝ (1, 1)</div>

For λ = 2: [[2,1],[2,1]]**v** = **0** gives 2v₁ = −v₂, so **v**₂ ∝ (1, −2).
Check: A(1,1) = (5,5) = 5(1,1) ✓.

Note det(A) = 10 = 5 × 2 and trace(A) = 7 = 5 + 2. Both hold in general —
**the determinant is the product of the eigenvalues and the trace is their
sum** — which gives you a free sanity check on any eigenvalue computation.

### Power iteration, and why it converges

Repeatedly multiply and renormalise. Why does it find the dominant
eigenvector? Write the start vector in the eigenbasis, **x**₀ = Σᵢ cᵢ**v**ᵢ.
Then

<div class="eq">A<sup>k</sup><b>x</b>₀ = Σᵢ cᵢλᵢ<sup>k</sup><b>v</b>ᵢ = λ₁<sup>k</sup>[ c₁<b>v</b>₁ + Σ<sub>i&gt;1</sub> cᵢ(λᵢ⁄λ₁)<sup>k</sup><b>v</b>ᵢ ]</div>

Every ratio |λᵢ/λ₁| < 1, so every term except the first decays geometrically.
**The convergence rate is |λ₂/λ₁|** — fast when the dominant eigenvalue is
well separated, painfully slow when λ₂ ≈ λ₁. ∎

```python
"""Power iteration for the dominant eigenpair of a symmetric matrix."""

import numpy as np


def power_iteration(a: np.ndarray, iters: int = 5000,
                    tol: float = 1e-13) -> tuple:
    """Finds the dominant eigenpair.

    Args:
        a: Symmetric matrix of shape (n, n).
        iters: Maximum number of iterations.
        tol: Stop when the iterate moves less than this.

    Returns:
        A tuple (eigenvalue, eigenvector) with the vector normalised.
    """
    rng = np.random.default_rng(0)
    v = rng.normal(size=a.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        w = a @ v
        norm = np.linalg.norm(w)
        if norm < 1e-300:
            return 0.0, v
        w /= norm
        if np.linalg.norm(w - v) < tol or np.linalg.norm(w + v) < tol:
            v = w
            break
        v = w
    return float(v @ a @ v), v          # Rayleigh quotient
```

The returned value **v**ᵀA**v** (for unit **v**) is the **Rayleigh quotient**,
and the largest eigenvalue is exactly its maximum over the unit sphere
*[MML p.207, Thm 7.2.3]*.

### Why the maximum of the Rayleigh quotient is λ_max

This is the fact that makes PCA an optimisation problem with a closed-form
answer, so it deserves a proof. Let A be symmetric with orthonormal
eigenvectors **u**ᵢ and eigenvalues λ₁ ≥ ⋯ ≥ λₙ. Write any unit **x** as
Σᵢ xᵢ**u**ᵢ, where Σᵢ xᵢ² = 1. Then

<div class="eq"><b>x</b>ᵀA<b>x</b> = Σᵢ xᵢ²λᵢ ≤ λ₁ Σᵢ xᵢ² = λ₁</div>

with equality when **x** = **u**₁. So the maximum is λ₁, attained at the top
eigenvector. ∎ Restricting **x** to be orthogonal to **u**₁ repeats the
argument one level down and gives λ₂ — which is why principal components come
out in order.

## 5.2 The spectral theorem

Every real **symmetric** matrix can be diagonalised by an orthogonal matrix
*[MML p.203]*:

<div class="eq">A = QΛQᵀ, &nbsp; QᵀQ = I, &nbsp; Λ = diag(λ₁, …, λₙ)</div>

Two parts of this are worth proving, because both are used constantly.

**Eigenvalues are real.** Suppose A**v** = λ**v** with possibly complex λ and
**v**. Take the conjugate transpose: **v**\*A = λ̄**v**\*. Then

<div class="eq"><b>v</b>*A<b>v</b> = λ<b>v</b>*<b>v</b> &nbsp;&nbsp;and&nbsp;&nbsp; <b>v</b>*A<b>v</b> = λ̄<b>v</b>*<b>v</b></div>

Since **v**\***v** > 0, λ = λ̄, so λ is real. ∎

**Eigenvectors of distinct eigenvalues are orthogonal.** With A**u** = λ**u**
and A**w** = μ**w**, λ ≠ μ:

<div class="eq">λ⟨<b>u</b>,<b>w</b>⟩ = ⟨A<b>u</b>,<b>w</b>⟩ = ⟨<b>u</b>,A<b>w</b>⟩ = μ⟨<b>u</b>,<b>w</b>⟩</div>

so (λ − μ)⟨**u**,**w**⟩ = 0, and since λ ≠ μ the inner product vanishes. ∎

This matters more than it looks, because the matrices that actually arise in
machine learning are symmetric almost without exception: covariance matrices,
Gram matrices, kernel matrices, Hessians. **A symmetric matrix is just a
scaling along orthogonal axes, viewed from a rotated frame.**

## 5.3 The singular value decomposition

Non-square, non-symmetric matrices — real data — get the SVD. The derivation
in *[MML p.207]* is pure wishful thinking, and it is worth following because
it shows where the pieces come from. Suppose

<div class="eq">A = UΣVᵀ</div>

with U ∈ ℝ^(n×n), V ∈ ℝ^(m×m) orthogonal and Σ ∈ ℝ^(n×m) diagonal. Then

<div class="eq">AAᵀ = UΣVᵀVΣᵀUᵀ = UΣ²Uᵀ &nbsp;&nbsp;and&nbsp;&nbsp; AᵀA = VΣ²Vᵀ</div>

Both are symmetric, so the spectral theorem hands them to us: **U** is the
eigenvectors of AAᵀ, **V** the eigenvectors of AᵀA, and the **singular values**
σᵢ are the square roots of their shared eigenvalues — non-negative, because
AᵀA is positive semi-definite (**x**ᵀAᵀA**x** = ‖A**x**‖² ≥ 0). The
decomposition exists for *every* matrix *[MML p.208, Thm 7.3.1]*.

### Worked example — the SVD of a 2×2

Take A = [[3, 0], [4, 5]]. Then

<div class="eq">AᵀA = [[25, 20], [20, 25]]</div>

whose eigenvalues solve (25−λ)² = 400, giving λ = 45 and λ = 5. So the
singular values are σ₁ = √45 ≈ 6.708 and σ₂ = √5 ≈ 2.236. The eigenvectors of
AᵀA are (1,1)/√2 and (1,−1)/√2 — these are the columns of V.

Check against the determinant: |det A| = 15, and σ₁σ₂ = √45·√5 = √225 = 15 ✓.
**The product of the singular values is the absolute determinant**, which is
the volume statement of Chapter 3 restated in the SVD's language.

Geometrically every linear map is therefore: **rotate (Vᵀ), scale along axes
(Σ), rotate again (U)**. Nothing else is possible. A is sending the unit
circle to an ellipse with semi-axes 6.708 and 2.236.

## 5.4 Low rank, and why it is the whole game

Truncate to the k largest singular values and you get the best rank-k
approximation that exists, in both the Frobenius and spectral norms — the
**Eckart–Young theorem**. The error is exactly the norm of what you discarded:

<div class="eq">‖A − A<sub>k</sub>‖<sub>F</sub> = √(σ²<sub>k+1</sub> + ⋯ + σ²<sub>r</sub>)</div>

### Why the error takes that form

The Frobenius norm is unitarily invariant: ‖UMVᵀ‖_F = ‖M‖_F, because
orthogonal matrices preserve lengths. So

<div class="eq">‖A − A<sub>k</sub>‖<sub>F</sub> = ‖U(Σ − Σ<sub>k</sub>)Vᵀ‖<sub>F</sub> = ‖Σ − Σ<sub>k</sub>‖<sub>F</sub></div>

and Σ − Σₖ is the diagonal matrix holding exactly the discarded singular
values, whose Frobenius norm is the root-sum-of-squares of them. ∎ (That the
truncation is *optimal* among all rank-k matrices is the harder half of
Eckart–Young; the error identity above is the half you will verify
numerically.)

```python
"""Low-rank approximation and the Eckart-Young error identity."""

import numpy as np


def svd_lowrank(a: np.ndarray, k: int) -> np.ndarray:
    """Returns the best rank-k approximation of a matrix.

    Args:
        a: Matrix to approximate.
        k: Target rank.

    Returns:
        The rank-k approximation, with the same shape as `a`.

    Raises:
        ValueError: If k is not in [1, min(a.shape)].
    """
    if not 1 <= k <= min(a.shape):
        raise ValueError(f'k must lie in [1, {min(a.shape)}], got {k}')
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return (u[:, :k] * s[:k]) @ vt[:k, :]


rng = np.random.default_rng(0)
a = rng.normal(size=(60, 30))
s = np.linalg.svd(a, compute_uv=False)
for k in (1, 5, 10, 20):
    predicted = float(np.sqrt(np.sum(s[k:] ** 2)))
    actual = float(np.linalg.norm(a - svd_lowrank(a, k)))
    print(f'k={k:>3}  predicted {predicted:9.5f}  actual {actual:9.5f}  '
          f'match {abs(predicted - actual) < 1e-10}')
```

That identity is not a curiosity. It is the reason compression works, the
reason PCA keeps what it keeps, the reason embeddings have the dimensions they
do, and the reason LoRA can fine-tune a billion-parameter model by training
two thin matrices.

## 5.5 PCA, derived three ways

PCA is usually presented as a recipe. It is better understood as the *same
answer* arrived at from three directions — and seeing that is what makes it
stop being a black box.

**(a) Maximum variance.** Find the unit direction **w** maximising the
variance of the projected data. With centred X, the projected variance is

<div class="eq">Var(X<b>w</b>) = (1⁄(n−1))‖X<b>w</b>‖² = <b>w</b>ᵀ C <b>w</b>, &nbsp; C = XᵀX⁄(n−1)</div>

Maximise **w**ᵀC**w** subject to ‖**w**‖ = 1. Lagrange:

<div class="eq">∇[<b>w</b>ᵀC<b>w</b> − λ(<b>w</b>ᵀ<b>w</b> − 1)] = 2C<b>w</b> − 2λ<b>w</b> = <b>0</b> &nbsp;⟹&nbsp; C<b>w</b> = λ<b>w</b></div>

The stationary points are the **eigenvectors of the covariance matrix**, and
by Section 5.1's Rayleigh argument the maximum is the top one. ∎

**(b) Minimum reconstruction error.** Find the k-dimensional subspace
minimising Σᵢ‖**x**ᵢ − proj(**x**ᵢ)‖². Since ‖**x**‖² = ‖proj‖² + ‖residual‖²
(Pythagoras, because the projection is orthogonal), minimising the residual is
*identical* to maximising the projected variance. Same problem, same answer.

**(c) The SVD.** Centre, take X = UΣVᵀ. Then C = XᵀX/(n−1) = VΣ²Vᵀ/(n−1), so
V holds the eigenvectors of C and the explained variance of component i is
σᵢ²/(n−1). Computing the SVD of X directly is numerically better than forming
C first, for exactly the κ(XᵀX) = κ(X)² reason from Chapter 4.

```python
def pca(x: np.ndarray, n_components: int) -> tuple:
    """Computes PCA directly from the SVD of the centred data.

    Args:
        x: Data matrix of shape (n_samples, n_features).
        n_components: Number of components to keep.

    Returns:
        A tuple (scores, components, explained_variance_ratio).
    """
    centred = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    variance = s ** 2 / (x.shape[0] - 1)
    scores = (u[:, :n_components] * s[:n_components])
    return scores, vt[:n_components], variance[:n_components] / variance.sum()
```

Two consequences people learn the hard way. **Centring is not optional** —
derivation (a) assumed it when it wrote the variance as **w**ᵀC**w**; skip it
and the first component points at the mean. And **scale matters**: PCA
maximises variance, and variance has units, so a feature in millimetres will
dominate the same feature in metres. Standardise unless the units are
genuinely comparable.

## 5.6 Gate for this chapter

| Check | Threshold |
|---|---|
| Your hand-computed eigenvalues of [[4,1],[2,3]] | 5 and 2 exactly |
| trace = Σλ and det = Πλ | to 1e-12 |
| Your hand-computed singular values of [[3,0],[4,5]] | √45 and √5 |
| `power_iteration` against `np.linalg.eigvalsh().max()` | relative 1e-6 |
| ‖A − QΛQᵀ‖ for symmetric A | < 1e-10 |
| Eckart–Young identity at k = 1, 5, 10, 20 | < 1e-10 absolute |
| Your PCA against the covariance eigendecomposition | ratios agree to 1e-8 |
