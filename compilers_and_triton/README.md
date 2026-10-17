# Compilers and Triton

A ladder from "what is a token" to a tensor compiler that emits fused
Triton kernels. Eight stages, each a single runnable file that prints
its own gates.

```bash
cd compilers_and_triton
python run_all.py            # every stage, ~8 seconds
python run_all.py --track a  # the compiler track only
```

On a CPU-only machine: **5 verified, 3 unverified, 0 failed**. The
three unverified are the GPU kernels — their reference implementations
are still checked, so nothing is silently skipped.

---

## Why the two tracks are one course

Most material treats "writing a compiler" and "writing GPU kernels" as
separate subjects. They are the same subject seen from two ends, and
the capstone is where that becomes concrete:

> A tensor compiler is a compiler. It parses, builds an intermediate
> representation, optimises it, and emits code. PyTorch's graph
> capture, Triton's lowering to PTX and XLA's fusion are the machinery
> of track A applied to the kernels of track B.

The capstone takes `max(x * w + b, 0)`, lowers it with the **front end
from A1–A2 unchanged**, optimises it with the **passes from A3
unchanged**, and emits one fused Triton kernel — three loads, one
store, instead of three kernel launches and six memory passes.

That the IR did not need changing to go from scalars to tensors is the
entire argument for having an IR.

---

## The ladder

### Track A — Compilers (runs anywhere, CPU only)

| Stage | Level | What you build |
|---|---|---|
| **A1** `a1_lexer_parser.py` | Fundamental | A lexer and a Pratt parser. Precedence as *data* rather than as a tower of recursive functions, so adding an operator is one line. |
| **A2** `a2_ir.py` | Fundamental | Lowering to three-address code in single-assignment form. `let` disappears — scoping was a front-end concern. The interpreter here is the specification everything downstream must preserve. |
| **A3** `a3_passes.py` | Intermediate | Constant folding, copy propagation, CSE and dead-code elimination, run to a fixpoint. `2*3+4*5-6/2` collapses from 11 instructions to 1. |
| **A4** `a4_schedule.py` | Advanced | Algorithm versus schedule. A readable cost model, a search over 344 feasible tilings, and a 120× reduction in memory traffic with the arithmetic untouched. |

### Track B — Triton (needs a GPU; use Colab)

| Stage | Level | What you build |
|---|---|---|
| **B1** `b1_vector_add.py` | Fundamental | The three ideas in every kernel: you write for a *block*, `program_id` says which one you are, and masks handle the ragged edge without branching. |
| **B2** `b2_softmax.py` | Intermediate | Reductions, and why every softmax subtracts the max. The naive version returns NaN on logits of 100 — demonstrated, not asserted. |
| **B3** `b3_matmul.py` | Advanced | The kernel A4's search described: square tiles, k innermost, accumulator in registers for the whole k loop, grouped ordering for L2 reuse. |

### Capstone

| | |
|---|---|
| **`capstone/tensor_compiler.py`** | Expression → front end → passes → fused Triton **and** fused NumPy, with both verified against a naive evaluation. Two backends from one IR; a new target costs a table and a prologue. |

---

## What the gates check

Every stage declares its gates before it runs. Some are worth
describing, because they encode the lessons:

- **A3** uses **differential testing**: each optimised program is run
  against the original on 200 random inputs. Proving a pass correct is
  hard; running it on hundreds of inputs is easy, and it is how real
  compilers are tested.
- **A3** also checks that `1 / 0` is **not** folded. Folding it would
  turn a runtime error into a compile-time crash — a change in
  behaviour, not an optimisation.
- **A4** checks the cost model reproduces three facts a performance
  engineer would know independently: tiling beats the naive loop,
  square tiles beat thin ones, and k innermost beats k outermost. A
  model that gets those wrong is not worth searching with.
- **B2** checks the naive softmax *does* produce NaN, so the lesson is
  demonstrated rather than claimed.
- **The capstone** checks the emitted Triton loads each input exactly
  once, and that an unsupported operation is **refused** rather than
  mis-emitted. A compiler that silently emits wrong code for an
  operation it does not understand is worse than one that crashes.

---

## The notebooks

One per stage, in `notebooks/`, each self-contained — no clone, no
install, no data to fetch. The module source is embedded, so a
notebook works in a fresh Colab runtime with nothing beside it.

| Stage | Open in Colab |
|---|---|
| A1 lexer and parser | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/treaper33-hue/ML-AI-assisted-/blob/main/compilers_and_triton/notebooks/a1_lexer_parser.ipynb) |
| A2 IR and interpreter | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/treaper33-hue/ML-AI-assisted-/blob/main/compilers_and_triton/notebooks/a2_ir.ipynb) |
| A3 optimisation passes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/treaper33-hue/ML-AI-assisted-/blob/main/compilers_and_triton/notebooks/a3_passes.ipynb) |
| A4 scheduling and autotuning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/treaper33-hue/ML-AI-assisted-/blob/main/compilers_and_triton/notebooks/a4_schedule.ipynb) |
| B1 vector add · **GPU** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/treaper33-hue/ML-AI-assisted-/blob/main/compilers_and_triton/notebooks/b1_vector_add.ipynb) |
| B2 softmax · **GPU** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/treaper33-hue/ML-AI-assisted-/blob/main/compilers_and_triton/notebooks/b2_softmax.ipynb) |
| B3 tiled matmul · **GPU** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/treaper33-hue/ML-AI-assisted-/blob/main/compilers_and_triton/notebooks/b3_matmul.ipynb) |
| Capstone tensor compiler | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/treaper33-hue/ML-AI-assisted-/blob/main/compilers_and_triton/notebooks/capstone_tensor_compiler.ipynb) |

Each notebook runs the stage and then takes it apart: four or five
exploration cells that change a parameter and show the consequence.
A1 demotes `*` below `+` and reparses. A4 reruns the search at six
cache budgets and the winner moves. The capstone deletes an entry from
the emission table and the compiler refuses the program.

The notebooks are **generated from the modules**, which are the single
source of truth, and the prose lives apart from the machinery:

```bash
cd compilers_and_triton/notebooks
python make_notebooks.py        # rebuild all eight
python validate_notebooks.py    # execute every cell of every one
```

`validate_notebooks.py` runs each notebook top to bottom in a real
module namespace — the same thing Colab does — and checks that setup
prints nothing, that the run cell reports a status, and that **every**
exploration cell executes. It has already earned its place: it caught
an exploration cell that gave `^` a binding power without teaching the
lexer the character, so the parser never saw a `^` at all. That cell
now does both edits and says why there are two.

### Running the GPU stages

Triton's official wheels are Linux-only, and installing a Windows port
beside a working PyTorch risks breaking it. The reliable route is
Colab, where Triton ships with PyTorch:

1. Open one of the three **GPU** notebooks above
2. **Runtime → Change runtime type → T4 GPU**
3. Run the cells

Each module reports its environment honestly at the top, and a kernel
that could not run is reported as `SKIP`. A skip is not a pass.

---

## Code contract, stated rather than claimed

Checked with the repository's own ruff configuration:

```bash
python -m ruff check --config testbed/pyproject.toml \
  --select E501,E731,F401,F541,F841,B023 compilers_and_triton/
```

That passes. It found six real defects on the first run, all now
fixed: a leftover `results` list in the capstone, an over-long line in
the environment banner, a lambda assigned to `grid`, and three
closures over loop variables in the kernel checks — harmless there,
because `check` calls them within the same iteration, but the tensors
are now bound as defaults so the code says so instead of a comment.

Three categories are **deliberately not clean**, and saying which is
more useful than a green tick:

- **`N803`** on kernel arguments — `M`, `N`, `K`, `BLOCK_SIZE`,
  `GROUP_M`. Triton's `tl.constexpr` parameters are capitalised by
  convention throughout its own documentation and kernels. Renaming
  them to satisfy a linter would make every tutorial harder to follow.
- **`E402`** on the `sys.path` lines — a module imports its sibling
  after putting its own directory on the path, which is the price of
  stages that run as plain scripts with no package install. Each is
  marked `# noqa: E402`.
- **`ANN`** on `self` and some arguments. Note that `testbed/` reports
  157 findings in the same categories under the same configuration, so
  this is the house standard rather than a lapse — but the top-level
  README's "Code contract" row says ruff is *clean*, and under current
  ruff that is no longer true of `testbed/` either. Worth correcting
  there.

## Where this sits

This section assumes the deep-learning material in the main programme
and extends tiers 8 and 9 of *ML Learning Volume II* — GPU, kernels and
Triton, then compilers and code generation. The reading there is the
theory; this is the code.

The natural next steps, none of them built here:

- **An autograd layer over the capstone**, so a fused kernel has a
  fused backward pass. This is what `torch.compile` does and where most
  of its speedup comes from.
- **Reductions in the emitted code.** The capstone fuses elementwise
  operations only; fusing a reduction into the same kernel is the step
  from "useful" to "what production compilers do".
- **A real benchmark loop** replacing A4's cost model, which is what
  `triton.autotune` is: the same search against the clock instead of a
  model.
