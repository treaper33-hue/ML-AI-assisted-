<div class="cover">
<div class="rule"></div>
<h1>Machine Learning<br>The Working Text</h1>
<div class="sub">The mathematics, the code, and the discipline —<br>from first principles to professional capability</div>
<div class="meta">
Built from <b>Mathematics of Machine Learning</b> (Danka) · <b>Prompting Scikit-Learn for Machine Learning</b><br>
<b>PyTorch: The Practical Guide</b> (Gollnick) · <b>Time Series with PyTorch</b> (Davidson &amp; Ma)<br>
<b>Google Python Style Guide</b> · the 30 foundational papers in <tt>papers/01..30</tt><br><br>
Every derivation is worked. Every code listing runs.<br>
Every claim carries the page it came from.
</div>
</div>

<!--TOC-->

# How to use this text {: .nobreak }

This is not a summary of four books. It is the material itself — the
derivations, the algorithms, the code, and the judgement — written out so you
can learn from this page rather than being sent somewhere else. Where a source
goes deeper than a working text should, you will find a marker like
*[MML p.207]*, telling you exactly which page to open and why.

**Read it with your hands.** Every chapter ends in code that runs. Type it;
do not paste it. The difference between a reader and a practitioner is
measured in keystrokes. Part VI then puts the pieces together into four
complete systems you can run in a minute, in `ml-mastery/projects/`.

**Three rules hold this text together.**

1. *Implement before you use.* Everything in Part I exists in NumPy already.
   You write it once yourself, then you use the library version forever with
   an informed conscience.
2. *One number, one source.* Metrics are defined once, in code, and every
   comparison in the book calls that one implementation. Three of the source
   books define overlapping metrics that disagree at the edges; those
   disagreements are resolved here, with the edge cases as tests.
3. *A result without a gate is not a result.* Each chapter states what must be
   numerically true for you to move on. Not "I understood it" — a number,
   within a tolerance, that you can show somebody.

## The sources, and what each one is for

| Tag | Source | What it is best at |
|---|---|---|
| **[MML]** | *Mathematics of Machine Learning*, Tivadar Danka, Packt 2025 | The mathematics, with proofs and intuition that actually connect. Printed page numbers. |
| **[SKL]** | *Prompting Scikit-Learn for Machine Learning* | Leak-safe pipelines, evaluation discipline, the workflow around a model. Chapter + section anchors. |
| **[PTG]** | *PyTorch: The Practical Guide*, Bert Gollnick, Rheinwerk 2026 | PyTorch mechanics end to end, through to deployment. Printed page numbers. |
| **[TSP]** | *Time Series with PyTorch*, Davidson & Ma, Packt 2026 | Temporal data, honest evaluation, uncertainty. PDF page indices. |
| **[GPSG]** | *Google Python Style Guide* | How the code should look when somebody else reads it. |
| **[30u30]** | `papers/01..30` in this repository | Thirty foundational papers, implemented. |

## Notation

| Symbol | Meaning |
|---|---|
| x, y | scalars |
| **x**, **y** | column vectors; xᵢ is the i-th entry |
| A, B | matrices; Aᵀ is the transpose; aᵢⱼ the entry in row i, column j |
| ℝⁿ | the space of real n-tuples |
| ‖**x**‖₂ | Euclidean norm, √(Σ xᵢ²) |
| ⟨**x**, **y**⟩ | inner product, Σ xᵢyᵢ |
| ∇f | gradient of f, the vector of partial derivatives |
| h | learning rate (the step size in gradient descent), following [MML p.383] |
| 𝔼[X] | expected value of the random variable X |
| p, q | probability distributions |
| θ | a model's parameters, collectively |
| L(θ) | the loss as a function of parameters |

Code follows the Google Python Style Guide: 80 columns, four-space indents,
type annotations on public functions, Google-style docstrings. Inside a
derivation, short mathematical names (`A`, `x`, `lr`) are used deliberately —
the style guide permits local names that match the mathematics
*[GPSG §3.16.5]*.

## The environment

Everything here runs on NumPy; Parts II–IV add scikit-learn, PyTorch and
statsmodels. The testbed that accompanies this text brings all of it up with
one command:

```powershell
cd ml-mastery\testbed
.\setup.ps1
```

If a package is missing, the experiments say so and mark the affected result
`UNVERIFIED` rather than skipping it quietly. That behaviour is the whole
philosophy of this text in miniature: **the absence of a check is itself
reported.**
