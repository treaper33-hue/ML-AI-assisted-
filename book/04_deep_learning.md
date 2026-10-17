# Part III — Neural Networks in PyTorch

Everything in this part is Part I applied at scale. The tensors are Chapter 1's
arrays with a gradient tape attached; autograd is Chapter 7's chain rule run
in reverse; the optimiser is Chapter 6's update rule; the loss is Chapter 8's
negative log-likelihood. If any of those four feel shaky, go back — this part
will be mechanical if they are solid and mystifying if they are not.

---

# 16 · Tensors and automatic differentiation

## 16.1 A tensor is an array that remembers

A PyTorch tensor is a typed multi-dimensional array on some device
*[PTG p.42–43]*. Set `requires_grad=True` and it gains something NumPy does
not have: every operation records itself onto a graph, so the chain rule can
later be replayed backwards *[PTG p.45]*.

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 2 * x                   # y = x³ + 2x,  dy/dx = 3x² + 2
y.backward()
print(x.grad)                        # 14.0 at x = 2
```

Three rules govern the graph, and almost every beginner bug is one of them:

1. **The graph is built during the forward pass and consumed by `backward()`.**
   Calling `backward()` twice needs `retain_graph=True`.
2. **Gradients accumulate.** `.grad` is added to, not overwritten. This is what
   makes gradient accumulation over micro-batches possible, and it is why
   `optimizer.zero_grad()` is mandatory — forget it and you are descending on
   the sum of every batch you have ever seen.
3. **Only leaves keep gradients.** Intermediate results drop theirs unless you
   ask with `retain_grad()`.

Detach from the graph when you do not want a path: `.detach()`, or a
`with torch.no_grad():` block for inference and evaluation. Failing to do so
in an eval loop is the most common cause of a memory leak that looks like a
hardware problem — the graph for every evaluated batch is retained, because
something still references it.

## 16.2 What `backward()` actually computes

A subtlety worth understanding, because it explains an error you will hit.
Autograd does not build the full Jacobian; it computes a **vector–Jacobian
product**, vᵀJ. For a scalar loss, v = 1 and the product is the gradient,
which is why `loss.backward()` works with no argument. Call `backward()` on a
non-scalar and PyTorch raises, because it does not know which v you meant —
`y.backward(torch.ones_like(y))` supplies it explicitly.

This is the cost argument of Chapter 7.2 realised in software: by only ever
propagating row vectors leftwards through the chain, reverse mode never
materialises an (output × parameter) Jacobian.

## 16.3 Prove autograd to yourself once

```python
"""Check PyTorch's gradients against finite differences."""

import torch


def check_gradient(f, x: torch.Tensor, h: float = 1e-4) -> float:
    """Returns the largest gap between autograd and central differences.

    Args:
        f: Scalar-valued function of a tensor.
        x: Point of evaluation, requires_grad not needed.
        h: Step size for the numerical derivative.

    Returns:
        The maximum absolute difference across coordinates.
    """
    x = x.detach().double().requires_grad_(True)
    f(x).backward()
    analytic = x.grad.clone()
    numeric = torch.zeros_like(x)
    with torch.no_grad():
        for i in range(x.numel()):
            step = torch.zeros_like(x)
            step.view(-1)[i] = h
            numeric.view(-1)[i] = (f(x + step) - f(x - step)) / (2 * h)
    return float((analytic - numeric).abs().max())


fn = lambda t: (torch.sin(t) * torch.exp(-t ** 2)).sum()
print('max gap:', check_gradient(fn, torch.randn(5)))   # ~1e-10
```

Chapter 6's error analysis applies here too: use h around 1e-4 to 1e-6 and
**double** precision. Gradient checking in float32 produces false alarms,
because ε ≈ 10⁻⁷ puts the round-off floor above the discrepancy you are
hunting.

---

# 17 · Layers, activations, and losses

## 17.1 Activations and their derivatives

The derivative matters as much as the function, because Chapter 7 showed it
multiplies into every gradient flowing backwards through that unit
*[PTG p.35–37]*.

| Activation | f(z) | f′(z) | Notes |
|---|---|---|---|
| **ReLU** | max(0, z) | 1 if z > 0 else 0 | Default. No saturation for z > 0, so gradients survive depth. Dead units if the input is always negative. |
| **Leaky ReLU** | max(αz, z) | 1 or α | Fixes dead units at the cost of one parameter. |
| **Sigmoid** | 1/(1+e⁻ᶻ) | f(z)(1−f(z)) | **Peaks at 0.25** (at z = 0). Output layers only. |
| **Tanh** | (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ) | 1 − f(z)² | Zero-centred, peaks at 1. Still saturates. |
| **GELU** | z·Φ(z) | smooth | The transformer default. |

Two of those derivatives are worth deriving, as both reappear constantly.

**Sigmoid.** σ(z) = (1 + e^{−z})^{−1}, so

<div class="eq">σ′(z) = e<sup>−z</sup>⁄(1 + e<sup>−z</sup>)² = σ(z)·(1 − σ(z))</div>

Its maximum is at z = 0 where σ = ½, giving σ′ = **0.25** — the vanishing
gradient coefficient from Chapter 7.

**Tanh.** tanh = (e^z − e^{−z})/(e^z + e^{−z}); differentiating the quotient
gives 1 − tanh². Maximum 1 at z = 0, four times better than sigmoid, which is
why tanh was preferred in RNNs long before ReLU existed.

## 17.2 Softmax and its Jacobian

**Softmax** turns a vector of scores into a distribution *[PTG p.37]*:

<div class="eq">softmax(<b>z</b>)ᵢ = e<sup>zᵢ</sup> ⁄ Σ<sub>j</sub> e<sup>z<sub>j</sub></sup></div>

Never implement it naively. e^z overflows for z ≳ 710 in float64 and ≳ 88 in
float32. Subtract the maximum first — the result is mathematically identical
because the constant cancels top and bottom:

```python
def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
```

Its Jacobian has a tidy closed form. Writing pᵢ for the i-th output:

<div class="eq">∂pᵢ⁄∂z<sub>j</sub> = pᵢ(δ<sub>ij</sub> − p<sub>j</sub>)</div>

**Derivation.** For i = j, differentiate the quotient: the numerator
contributes e^{zᵢ}·S and the denominator −e^{zᵢ}·e^{zᵢ}, over S², giving
pᵢ − pᵢ² = pᵢ(1 − pᵢ). For i ≠ j only the denominator depends on zⱼ, giving
−e^{zᵢ}e^{zⱼ}/S² = −pᵢpⱼ. ∎

## 17.3 Losses are likelihoods

Chapter 8 did the real work; this is the dictionary *[PTG p.38–40]*:

| Task | Loss | Assumption it encodes |
|---|---|---|
| Regression | MSE = (1/n)Σ(y − ŷ)² | Gaussian noise, constant variance |
| Regression, outliers | MAE / Huber | Laplace / mixed |
| Binary | BCE = −Σ[y log p + (1−y) log(1−p)] | Bernoulli |
| Multi-class | Cross-entropy = −Σ y log p | Categorical |

### The softmax + cross-entropy gradient, in full

This is the most important derivative in supervised deep learning, and the
cancellation is beautiful. With one-hot target **y** and L = −Σₖ yₖ log pₖ:

<div class="eq">∂L⁄∂z<sub>j</sub> = −Σ<sub>k</sub> (y<sub>k</sub>⁄p<sub>k</sub>)·∂p<sub>k</sub>⁄∂z<sub>j</sub> = −Σ<sub>k</sub> (y<sub>k</sub>⁄p<sub>k</sub>)·p<sub>k</sub>(δ<sub>kj</sub> − p<sub>j</sub>)</div>

The pₖ cancels:

<div class="eq">= −Σ<sub>k</sub> y<sub>k</sub>(δ<sub>kj</sub> − p<sub>j</sub>) = −y<sub>j</sub> + p<sub>j</sub>Σ<sub>k</sub> y<sub>k</sub> = p<sub>j</sub> − y<sub>j</sub></div>

using Σₖ yₖ = 1 for a one-hot target. So

<div class="eq">∂L⁄∂<b>z</b> = <b>p</b> − <b>y</b></div>

∎ — the same *prediction minus truth* as linear regression (Chapter 4) and
logistic regression (Chapter 12).

Two practical consequences follow directly from that derivation:

- **Use the fused loss.** `CrossEntropyLoss` and `BCEWithLogitsLoss` take
  *logits* and compute this directly via log-sum-exp, never forming p and then
  logging it. Applying softmax yourself and then taking a log is the classic
  route to NaN, because log(0) happens the moment a probability underflows.
- **The gradient never saturates.** Unlike squared error on a sigmoid output —
  where an σ′ factor of 0.25 or less survives — here the activation derivative
  cancelled entirely. A confidently wrong prediction produces a gradient of
  magnitude ≈ 1, so the network can recover from it.

## 17.4 Initialisation, derived

Initialise all weights to zero and every unit in a layer computes the same
thing and receives the same gradient forever — the network has one unit
pretending to be many. Initialise too large and activations saturate or
explode. The right scale follows from a variance calculation.

For z = Σᵢ wᵢxᵢ with independent, zero-mean w and x:

<div class="eq">Var(z) = n<sub>in</sub>·Var(w)·Var(x)</div>

To keep Var(z) = Var(x) through the layer we need Var(w) = 1/n_in. **Xavier /
Glorot** compromises between the forward and backward passes with
Var(w) = 2/(n_in + n_out), appropriate for tanh and sigmoid.

ReLU zeroes half its inputs, halving the variance, so the factor must be
doubled: **He / Kaiming**, Var(w) = 2/n_in. That factor of two is the entire
difference between the two schemes, and it is the difference between a
50-layer ReLU network training and dying.

PyTorch's defaults are sensible; override deliberately or not at all.

## 17.5 Normalisation

Both common normalisers subtract a mean and divide by a standard deviation,
then re-scale with learned parameters γ and β:

<div class="eq">x̂ = (x − μ)⁄√(σ² + ε), &nbsp;&nbsp; y = γx̂ + β</div>

The difference is **which axis** the statistics come from:

| | Statistics over | Depends on batch? | Train ≠ eval? | Used in |
|---|---|---|---|---|
| **BatchNorm** | the batch, per channel | yes | yes — running averages at eval | CNNs |
| **LayerNorm** | the features, per sample | no | no | transformers, RNNs |

BatchNorm's batch dependence is why small batches degrade it, and why it needs
separate train/eval behaviour — a frequent source of "my model scores
differently in eval mode" confusion. LayerNorm has neither problem, which is
why sequence models standardised on it.

---

# 18 · The training loop

## 18.1 Dataset and DataLoader

A `Dataset` answers two questions — how many items, and what is item i
*[PTG p.76]*. A `DataLoader` batches, shuffles, and parallelises the fetching.

```python
"""A minimal Dataset and the training loop with no framework helpers."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ArrayDataset(Dataset):
    """Wraps NumPy arrays as a PyTorch dataset.

    Attributes:
        x: Feature tensor of shape (n, d).
        y: Target tensor of shape (n, 1).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """Stores the arrays as float32 tensors.

        Args:
            x: Features of shape (n, d).
            y: Targets of shape (n,).

        Raises:
            ValueError: If the lengths differ.
        """
        if len(x) != len(y):
            raise ValueError(f'length mismatch: {len(x)} vs {len(y)}')
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.x)

    def __getitem__(self, index: int) -> tuple:
        """Returns the (features, target) pair at `index`."""
        return self.x[index], self.y[index]


def train(model, loader, epochs: int, lr: float) -> list:
    """Trains a model and returns the per-epoch mean loss.

    Args:
        model: A torch.nn.Module.
        loader: A DataLoader over the training set.
        epochs: Number of passes over the data.
        lr: Learning rate.

    Returns:
        A list of mean losses, one per epoch.
    """
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = []
    model.train()
    for _ in range(epochs):
        running, batches = 0.0, 0
        for xb, yb in loader:
            optimiser.zero_grad()        # 1. clear accumulated gradients
            prediction = model(xb)       # 2. forward
            loss = loss_fn(prediction, yb)
            loss.backward()              # 3. backward
            optimiser.step()             # 4. update
            running += float(loss)
            batches += 1
        history.append(running / batches)
    return history
```

Those four lines in the inner loop are the entire algorithm. Everything
PyTorch Lightning, Hugging Face Trainer and the rest provide is scaffolding
around them *[PTG p.351]* — and you should be able to write them from memory
before you let a framework write them for you.

## 18.2 Batch size

Batching trades gradient quality against throughput *[PTG p.72]*. The
mini-batch gradient is an unbiased estimator of the full gradient, and by
Chapter 8's variance result its standard error scales as

<div class="eq">SE ∝ σ⁄√B</div>

so quadrupling the batch halves the gradient noise — the same √n law, charging
its usual rate.

| Batch | Gradient | Hardware | Generalisation |
|---|---|---|---|
| 1 (SGD) | very noisy | idle GPU | noise acts as regularisation |
| 32–256 | good compromise | efficient | the usual default |
| Very large | near-exact | saturated | often worse; needs LR scaling and warm-up |

Because noise scales as 1/√B while the step size scales linearly, keeping the
*ratio* of noise to step constant when you grow the batch requires raising the
learning rate — the origin of the **linear scaling rule** (multiply the LR by
the same factor as the batch, with a warm-up to survive the early steps).

The noise in small-batch SGD is not purely a cost: it helps escape sharp
minima, which is why simply raising the batch size for speed, without touching
the learning rate, usually hurts.

## 18.3 Optimisers

| Optimiser | Update | When |
|---|---|---|
| SGD | w ← w − h·∇L | with momentum, still the strongest for vision |
| SGD + momentum | v ← βv − h∇L; w ← w + v | Chapter 6's heavy ball |
| Adam | per-parameter adaptive steps | the safe default elsewhere |
| AdamW | Adam with decoupled weight decay | the right Adam for regularised training |

### Adam, line by line

<div class="eq">m<sub>t</sub> = β₁m<sub>t−1</sub> + (1 − β₁)g<sub>t</sub> &nbsp;&nbsp;·&nbsp;&nbsp; v<sub>t</sub> = β₂v<sub>t−1</sub> + (1 − β₂)g<sub>t</sub>²</div>
<div class="eq">m̂<sub>t</sub> = m<sub>t</sub>⁄(1 − β₁<sup>t</sup>) &nbsp;&nbsp;·&nbsp;&nbsp; v̂<sub>t</sub> = v<sub>t</sub>⁄(1 − β₂<sup>t</sup>)</div>
<div class="eq">w<sub>t</sub> = w<sub>t−1</sub> − h·m̂<sub>t</sub>⁄(√v̂<sub>t</sub> + ε)</div>

m is a running mean of the gradient (momentum); v is a running mean of its
square (a per-parameter scale). Dividing by √v gives each parameter its own
effective step — which is why Adam is forgiving of the bad conditioning that
cripples plain SGD (Chapter 6).

**Why the bias correction?** m₀ = 0, so after one step m₁ = (1 − β₁)g₁, which
underestimates the gradient by a factor of (1 − β₁) = 0.1 at the usual
β₁ = 0.9. Taking expectations of the recursion gives 𝔼[mₜ] ≈ (1 − β₁ᵗ)𝔼[g], so
dividing by (1 − β₁ᵗ) removes the bias exactly. The correction matters most in
the first few dozen steps and decays to nothing — but without it, early
updates are badly scaled, which in a short fine-tune is the whole run.

In plain Adam, L2 weight decay enters through the gradient and is then divided
by √v, so strongly-updated parameters get *less* decay — the opposite of the
intent. **AdamW** applies the decay directly to the weights instead, which is
why it is standard for transformers.

**Schedules** matter more than people expect. Cosine decay with a short
warm-up is a reliable default. To choose the peak rate, run an **LR range
test** *[TSP ch.6 p.245]*: increase the rate geometrically over one epoch,
plot loss against rate, and take roughly an order of magnitude below the
minimum. You have seen that curve already in Chapter 6 — the divergence edge
is the 2/L edge.

## 18.4 Determinism and checkpoints

"It's random" is not an acceptable explanation for a run you cannot reproduce.

```python
def set_seed(seed: int) -> None:
    """Seeds every generator that affects a training run."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
```

Save the *state dict*, not the pickled model *[PTG p.80]*, and save enough to
resume rather than merely to infer: model state, optimiser state (Adam's m and
v are parameters too), scheduler state, epoch, and the seed. A checkpoint that
cannot resume training is half a checkpoint.

## 18.5 Reading a loss curve

| Symptom | Likely cause |
|---|---|
| Loss flat from step 0 | LR far too low, or gradients not reaching (missing `zero_grad`, detached graph) |
| Loss → NaN | LR above the stability limit, or log(0) from an unfused softmax |
| Loss explodes then recovers | one bad batch; clip gradients |
| Train falls, validation rises | overfitting — regularise, augment, or stop earlier |
| Both fall then plateau high | underfitting — more capacity, longer schedule, better features |
| Validation below training | not a miracle: dropout is active in training, or your split leaks |

---

# 19 · Convolution

## 19.1 Why not a dense layer

A 224×224 RGB image flattened is 150,528 inputs; one dense layer of 1,000
units is 150 million parameters, and it learns nothing about the fact that
neighbouring pixels are related. Convolution encodes two priors
*[PTG p.128–129]*:

- **Locality** — features depend on nearby pixels;
- **Translation equivariance** — an edge is an edge wherever it appears, so
  the same kernel applies everywhere.

Weight sharing turns those priors into a parameter count that does not scale
with image size.

## 19.2 The arithmetic you must know

<div class="eq">out = ⌊(in + 2·pad − dilation·(kernel − 1) − 1) ⁄ stride⌋ + 1</div>

### Worked example

A 32×32 input, 3×3 kernel, stride 1, padding 1:

<div class="eq">(32 + 2 − 1·(3−1) − 1)⁄1 + 1 = (32 + 2 − 2 − 1) + 1 = <b>32</b></div>

Size preserved — which is why odd kernels with pad = (k−1)/2 dominate. Same
input with stride 2:

<div class="eq">⌊(32 + 2 − 2 − 1)⁄2⌋ + 1 = ⌊15.5⌋ + 1 = <b>16</b></div>

Halved, which is how you downsample without pooling.

Parameters for C_in → C_out with k×k kernels: C_out·(C_in·k² + 1),
**independent of the input resolution**. A 3×3 layer from 64 to 128 channels
is 128·(64·9 + 1) = 73,856 parameters whether the image is 32×32 or 1024×1024.

The **receptive field** grows linearly with depth for stride-1 layers: two
stacked 3×3 convolutions see 5×5, three see 7×7 — with fewer parameters
(2·9 = 18 vs 49 per channel pair) and more nonlinearity than a single 7×7.
That observation is most of what VGG contributed, and it is implemented in
`papers/09_resnet` in this repository. Dilation grows the field
*exponentially* without pooling — `papers/11_dilated_convolutions`.

## 19.3 Convolution is a matrix multiply

Under the hood, `im2col` unfolds every patch into a row, turning convolution
into the GEMM that Chapter 3 described and that GPUs are built for:

```python
def conv2d_via_im2col(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Single-channel 2-D convolution expressed as a matrix product.

    Args:
        x: Input of shape (H, W).
        kernel: Kernel of shape (kh, kw).

    Returns:
        The valid cross-correlation, shape (H-kh+1, W-kw+1).
    """
    kh, kw = kernel.shape
    out_h, out_w = x.shape[0] - kh + 1, x.shape[1] - kw + 1
    patches = np.lib.stride_tricks.sliding_window_view(x, (kh, kw))
    cols = patches.reshape(out_h * out_w, kh * kw)
    return (cols @ kernel.ravel()).reshape(out_h, out_w)
```

Note that this, like PyTorch's `conv2d`, computes *cross-correlation* — the
kernel is not flipped. Since the kernel is learned, the distinction is
irrelevant to the result and matters only when comparing against a
signal-processing reference.

### The backward pass

Once convolution is a matrix multiply, Chapter 7's rules apply unchanged. With
Y = W·X_col:

<div class="eq">∂L⁄∂W = δ·X<sub>col</sub>ᵀ &nbsp;&nbsp;·&nbsp;&nbsp; ∂L⁄∂X<sub>col</sub> = Wᵀ·δ</div>

The second is then folded back into image shape (`col2im`), summing the
contributions of overlapping patches — which is itself a convolution with the
kernel flipped. **The gradient of a convolution is a convolution**, which is
why conv layers are as cheap backwards as forwards.

---

# 20 · Autoencoders and the variational autoencoder

## 20.1 Autoencoders

An autoencoder learns to reconstruct its input through a narrow bottleneck
*[PTG p.220]*. With linear layers and squared error it recovers the PCA
subspace of Chapter 5 — the same optimum, by a slower route, because
minimising reconstruction error over a k-dimensional linear subspace *is*
derivation (b) of Section 5.5. (It finds the subspace, not the orthonormal
basis: the solution is unique only up to an invertible k×k transform.)

The honest benchmark is therefore PCA at the same latent dimension. If your
autoencoder cannot beat it, it has learned nothing a 1901 algorithm did not —
and the fix is nonlinearity, not more epochs.

## 20.2 The VAE, derived

A variational autoencoder models the data as generated from a latent **z**
*[PTG p.230]*. We want log p(x) and cannot compute it, because it requires
integrating over every possible z. So bound it.

For *any* distribution q(z|x), start from the definition and insert q twice:

<div class="eq">log p(x) = log ∫ p(x,z) dz = log ∫ q(z|x)·(p(x,z)⁄q(z|x)) dz</div>

Jensen's inequality (Chapter 8, log is concave) moves the log inside:

<div class="eq">log p(x) ≥ ∫ q(z|x)·log(p(x,z)⁄q(z|x)) dz = 𝔼<sub>q</sub>[log p(x|z)] − D<sub>KL</sub>(q(z|x) ‖ p(z))</div>

That right-hand side is the **ELBO**. The gap between it and log p(x) is
exactly D_KL(q(z|x) ‖ p(z|x)) — non-negative by Gibbs, zero only when q is the
true posterior. So **maximising the ELBO does two things at once**: it pushes
up a lower bound on the likelihood, and it pushes q towards the true
posterior.

Two terms with plain meanings: **reconstruct the input well**, and **keep the
latent distribution close to the prior**. For Gaussian q and standard normal
prior the KL has a closed form:

<div class="eq">D<sub>KL</sub> = −½ Σ<sub>j</sub> (1 + log σ<sub>j</sub>² − μ<sub>j</sub>² − σ<sub>j</sub>²)</div>

### The reparameterisation trick

Sampling z ~ q blocks gradients: you cannot differentiate through a sampler
whose parameters determine the distribution being sampled. Rewrite the sample
as a deterministic function of the parameters and *external* noise:

<div class="eq">z = μ + σ ⊙ ε, &nbsp;&nbsp; ε ~ N(0, I)</div>

Now the randomness lives in ε, which carries no parameters, and ∂z/∂μ = 1,
∂z/∂σ = ε are both well-defined. The path from loss to encoder stays
differentiable. (The alternative, score-function estimators, work without this
trick but have variance orders of magnitude higher — which is why VAEs became
practical only with the reparameterisation.)

```python
def vae_loss(x_hat, x, mu, logvar):
    """ELBO as a minimisable loss: reconstruction plus KL.

    Args:
        x_hat: Reconstruction, shape (batch, d).
        x: Original input, shape (batch, d).
        mu: Latent means, shape (batch, k).
        logvar: Latent log-variances, shape (batch, k).

    Returns:
        A tuple (total, reconstruction, kl), each a scalar tensor.
    """
    recon = torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl
```

The network predicts `logvar` rather than σ because log-variance is
unconstrained — predicting σ directly would need a positivity constraint, and
exp() enforces it for free while keeping gradients well-scaled.

**Monitor both terms separately.** If the KL collapses to zero the encoder has
stopped using the latent space and you have an ordinary autoencoder wearing a
VAE's clothes. Watching only the sum hides this completely.

---

# 21 · Attention and transformers

## 21.1 Scaled dot-product attention

Every position builds a query, and every position offers a key and a value.
Relevance is a dot product — Chapter 2's inner product used as a similarity —
and the output is a weighted average of values *[PTG p.320]*:

<div class="eq">Attention(Q, K, V) = softmax(QKᵀ ⁄ √d<sub>k</sub>) · V</div>

### Why √d_k, derived

The scaling is not decoration. Suppose query and key entries are independent
with zero mean and unit variance. Their dot product is a sum of d_k such
products, so by Chapter 8's variance-of-a-sum:

<div class="eq">Var(<b>q</b>·<b>k</b>) = Σ<sub>i=1</sub><sup>d<sub>k</sub></sup> Var(q<sub>i</sub>k<sub>i</sub>) = d<sub>k</sub></div>

Standard deviation √d_k. At d_k = 64 that is ±8 before scaling, and a softmax
over logits spread ±8 is essentially a hard maximum — one entry near 1, the
rest near 0. From Section 17.2's Jacobian, pᵢ(δᵢⱼ − pⱼ) → 0 as p → one-hot, so
**the gradient vanishes**.

Dividing by √d_k restores unit variance, keeps the softmax in its responsive
range, and keeps gradients alive. ∎ One divided square root is the difference
between a trainable mechanism and a dead one.

**Multi-head** attention runs h of these in parallel on projected subspaces
and concatenates, letting different heads attend to different kinds of
relation — and since each head uses d_k = d/h, the total cost is roughly that
of single-head attention at full width.

Cost: QKᵀ is (n × n), so attention is **O(n²)** in sequence length, in time
and memory. That single fact drives context-length limits, and every
efficient-attention variant exists to attack it.

## 21.2 The rest of the block

| Component | Purpose |
|---|---|
| Positional encoding | attention is permutation-invariant; without this, "dog bites man" equals "man bites dog" |
| Residual connection | a gradient highway: ∂(x + f(x))/∂x = 1 + f′, so gradients survive depth — `papers/09_resnet` |
| Layer normalisation | stabilises scale per token; pre-norm placement trains more reliably at depth |
| Feed-forward (×4 width) | per-position processing; most of the parameters live here |
| Causal mask | in a decoder, prevents attending to the future — the architectural form of Chapter 10's lookforward leakage |

The residual entry deserves emphasis. Chapter 7 showed gradients die through
products of small factors; a residual branch contributes an **additive 1** to
that product, so the gradient has a path that cannot be attenuated. It is the
same trick as the LSTM's additive cell state (Chapter 24), arrived at
independently.

Implement one yourself before using `nn.TransformerEncoder`.
`papers/13_attention` and `papers/14_annotated_transformer` walk through it;
the gate is that your attention output matches PyTorch's to 1e-5 on the same
weights.

---

# Part III gate

| Check | Threshold |
|---|---|
| Autograd vs finite differences | < 1e-9 in double precision |
| Hand-written backprop vs autograd on a 2-layer net | < 1e-7 |
| Your softmax Jacobian vs numerical | < 1e-6 |
| Softmax + CE gradient equals p − y | < 1e-6 |
| Naive vs stable softmax at z = 1000 | naive overflows; stable does not |
| Conv output sizes vs the formula | exact at stride 1 and 2 |
| Your `conv2d_via_im2col` vs `F.conv2d` | < 1e-5 |
| Checkpoint reload reproduces logits | bit for bit |
| Autoencoder vs PCA at equal latent dim | compared, result reported either way |
| VAE run | reconstruction and KL logged separately |
| Your attention vs `nn.MultiheadAttention` | < 1e-5 |
