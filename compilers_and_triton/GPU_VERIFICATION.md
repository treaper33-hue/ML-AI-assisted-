# GPU verification — 2026-10-17

The three Triton stages and the capstone, run on real hardware for the
first time. Before this, `run_all.py` reported **5 verified, 3
unverified**; the three were the GPU kernels, checked only against
their NumPy references.

**Result: 8 verified, 0 unverified, 0 failed** in 10.0 s.

The run found **eight defects**, four of them in code that had passed
every CPU check and three in the installer written to reproduce the
run. That ratio is the argument for doing this: a reference
implementation agreeing with itself proves very little.

---

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, `sm_120` (Blackwell), 8151 MiB |
| Driver | 616.56 |
| PyTorch | 2.13.0, CUDA 13.4, `torch.cuda.is_available()` True |
| Triton | `triton-windows` 3.8.0.post29 (reports `triton.__version__` 3.8.0) |
| Compiler | MSVC 14.50.35717 (Visual Studio 18) |
| Python | 3.11.16 (conda env `qwen35`) |
| OS | Windows 11, 10.0.26200 |

### Why not WSL2

WSL2 was the intended route and is not available on this machine:

```
WSL2 is not supported with your current machine configuration.
Please enable the "Virtual Machine Platform" optional component and
ensure virtualization is enabled in the BIOS.
```

The WSL binaries are installed (2.6.3.0) and one distro is registered
(`docker-desktop`, stopped), but the platform component is off.
Enabling a Windows optional component and possibly a BIOS setting is
the machine owner's decision and needs a reboot, so it was not done.

### What was done instead

`triton-windows` was installed with `pip install --target` into a
scratch directory, **not** into the `qwen35` conda environment. That
environment's PyTorch had already needed repair once (BUG-001, two
stacked distributions), and nothing was written into it here: its
`site-packages` is untouched, and Triton reaches Python through
`PYTHONPATH` for the duration of a single process.

This is packaged as **[`gpu_setup.py`](gpu_setup.py)**, so it is
reproducible rather than described:

```bash
python gpu_setup.py --install      # ~158 MB, to ~/.ml-mastery-gpu
python gpu_setup.py --check        # compiles and launches one kernel
python gpu_setup.py --run run_all.py
python gpu_setup.py --remove
```

It supplies the three things a Triton kernel compile needs on Windows
— MSVC on `PATH` via `vcvars64.bat`, located through `vswhere.exe`;
the isolated package directory on `PYTHONPATH`; and `CUDA_PATH`.

**`CUDA_PATH` is not optional, and the reason is worth recording.**
Without it the link fails:

```
LINK : fatal error LNK1181: cannot open input file 'cuda.lib'
```

`triton-windows` warns "Failed to find CUDA. Install CUDA Toolkit" —
but no CUDA Toolkit is needed. The wheel already bundles `cuda.h` and
`lib/x64/cuda.lib`, in exactly the Toolkit's directory layout; it just
puts only `lib/` on the link path, not `lib/x64/`. Pointing
`CUDA_PATH` at the bundled `triton/backends/nvidia` directory makes
Triton add the right path itself. No toolkit download, no file copied.

Writing that script turned up three more defects, listed as 6, 7 and 8
below. Two of them were in the script itself and one was a repeat, by
me, of defect 4.

---

## Results per stage

| Stage | Verdict | Measured |
|---|---|---|
| B1 vector add | VERIFIED | max relative error **0.00e+00** at n = 1,024 / 100,000 / 1,000,003. Exactly zero, because float32 addition of two float32 values is exact — there is nothing to accumulate. |
| B2 softmax | VERIFIED | max relative error 9.29e-07, 1.02e-06, 9.50e-07 across 128x256, 1024x781, 64x2048. Row sums to 1 within 1.1e-16. |
| B3 matmul | VERIFIED **after two fixes** | normwise relative error 2.88e-07, 4.55e-07, 1.81e-07 — true float32, once the precision was made explicit. |
| Capstone | VERIFIED | emitted NumPy exact (0.0e+00) on all five expressions; fusion ratios as predicted. |

### A known gap that 8/8 does not close

The capstone's **emitted Triton is still only compiled as Python
text**, never launched. The gate checks that the generated source
parses and that it loads each input once; it does not run it. Closing
that means generating the launcher alongside the kernel, which is real
work and is the obvious next step for this section.

---

## Defects found

The first five came from running the kernels; 6 to 8 came from
packaging the setup so the run can be repeated.

### 1. B3's gate measured the wrong thing — the gate was wrong, not the kernel

B3 failed on all three shapes with "max relative error" of 1.71e+01,
1.34e+02 and 9.39e+00. Those are not precision numbers; they are a
metric collapsing.

`runtime.relative_error` divides **elementwise**, with a 1e-6 floor.
That is APE, and APE is unbounded at zeros. A matrix product of random
data straddles zero everywhere, so an absolute error of 1e-4 against
an entry that happens to be 7.7e-05 reads as 1.3e+02.

The decisive evidence is that the metric also rejects the reference
implementation of the operation:

| | absolute error | normwise | elementwise (APE) |
|---|---|---|---|
| Triton, 512x640 @ 640x384 | 8.9e-02 | 7.70e-04 | **1.34e+02** |
| `torch.matmul`, same shapes | 3.2e-05 | 2.07e-07 | **6.24e-02** |

A 1e-3 elementwise gate fails `torch.matmul`. A gate that rejects
cuBLAS is not measuring the kernel.

**Fixed** by adding `runtime.normwise_error` — the Frobenius-norm
relative error that numerical linear algebra uses for exactly this
reason — and a `metric` parameter to `check`, with B3 selecting
`'normwise'`. B1 and B2 keep the elementwise measure, and should: B1's
output is exact, and softmax outputs are positive and bounded away
from zero, so there is no denominator to blow up.

This is the same class of defect as the three project gates recorded in
the book's §30.4, §31.3 and §33.2 — the threshold was written down
before the problem was understood.

### 2. The "float32" kernel was multiplying in TF32

With the metric fixed, Triton scored 7.70e-04 normwise where
`torch.matmul` scored 2.07e-07 — a factor of 3,700. That is not
float32; a 10-bit mantissa is the signature of TF32.

Confirmed by making the precision explicit, 1024³:

| | normwise error | ms | TFLOP/s |
|---|---|---|---|
| `tl.dot` default | 7.72e-04 | 0.117 | 18.3 |
| `tl.dot` `input_precision='tf32'` | 7.72e-04 | 0.117 | 18.4 |
| `tl.dot` `input_precision='ieee'` | 5.73e-07 | 0.205 | 10.5 |
| `torch.matmul` (precision `highest`) | 3.34e-07 | 0.181 | 11.9 |

The default is bit-for-bit identical to `tf32`, in both error and
time. **`tl.dot` rounds float32 inputs to TF32 unless told
otherwise.**

The module's own docstring made this worse rather than better: it
explained at length why the accumulator is float32, which is true and
irrelevant here. The accumulator was never the problem — the rounding
happens *before* the multiply, and no accumulator precision can undo
it.

**Fixed** by threading `PRECISION` through as a `tl.constexpr`,
defaulting `matmul(...)` to `'ieee'` so a function returning float32
has multiplied in float32, and adding `precision_comparison()` with
two gates: that TF32 is measurably less accurate (demonstrated, not
asserted, in B2's style) and that `'ieee'` really reaches float32.

The trade is now visible rather than silent: **1,347x less accurate,
1.7x faster**. Reasonable for training a network, unreasonable for
solving a linear system — and the point is that it is a choice.

### 3. The benchmark statistic was wrong

`torch.matmul` timed 0.802 ms in one run and 0.177 ms in another, a
4.5x spread, because the timing took the **mean** of 20 launches. On a
laptop GPU, clocks move with temperature and with whatever else is
using the device, and every one of those effects makes a measurement
slower — so the mean of a noisy sample estimates the noise.

**Fixed** with `best_milliseconds()`: the minimum over five rounds of
twenty launches, which is the standard choice for the same reason.
Two consecutive runs now agree to within 5%.

### 4. The notebook validator was not faithful to Colab

All three GPU notebooks failed with:

```
ValueError: @jit functions should be defined in a Python file
```

The notebooks were fine. `@triton.jit` reads its function's source
with `inspect.getsourcelines`, which consults `linecache`; cell code
lives in no file, and IPython registers each cell under a
pseudo-filename, which is why Triton works in Colab at all. The
validator `exec`'d cells without that registration.

**Fixed** in `validate_notebooks.py`, which now registers each cell's
source exactly as IPython does. This is the second time this harness
has been the unfaithful party — the first was `dataclasses` needing a
real module in `sys.modules`. A validator's only value is fidelity to
the runtime it stands in for.

### 5. The tuning cell crashed on a configuration A4 called feasible

B3's notebook benchmark cell died on `128x128 k=64`:

```
OutOfResources: out of resource: shared memory,
Required: 131072, Hardware limit: 101376
```

**Fixed** by catching it and reporting it, because this is not noise —
it is A4's feasibility constraint arriving as a real error, and it is
worth more to a reader than a table with one row missing.

### 6. `cmd /c "... && set"` cannot survive the quoting layers

Capturing the MSVC environment means running `vcvars64.bat` and asking
the shell what changed, because there is no supported way to read it.
Passing that as `['cmd', '/c', f'call "{vcvars}" && set']` failed with
"The system cannot find the path specified": the path contains spaces
and parentheses, and both `subprocess`'s `list2cmdline` and `cmd`'s own
parser get a turn at the quoting.

**Fixed** by writing a two-line batch file to a temporary directory and
running that, which removes every quoting layer at once.

### 7. The default prefix landed inside an application's private cache

`LOCALAPPDATA` looked like the obvious place for a 158 MB binary
install. It is not: a packaged application gets a **redirected**
`LOCALAPPDATA`, so running the installer from inside one put the files
in `AppData\Packages\...\LocalCache`, where the user's own terminal
would never find them and a cache cleanup could delete them without
warning.

**Fixed** by defaulting to `~/.ml-mastery-gpu`, which is predictable
on every platform and is not redirected. The misplaced copy was
removed.

### 8. The installer's own probe repeated defect 4

The post-install check ran its test kernel with `python -c`, and got:

```
ValueError: @jit functions should be defined in a Python file
```

The same defect as 4, in new code, written after diagnosing it — code
passed as `-c` has no file for `inspect.getsourcelines` to read, so
`@triton.jit` refuses it. Knowing the cause did not stop me
reintroducing it, which is the argument for a check that runs the
thing rather than a reviewer who remembers.

**Fixed** by writing the probe to a temporary file.

---

## A4's cost model, marked against the clock

A4 ranks tilings by modelled DRAM traffic. The question is not whether
its byte counts are right — they are fiction — but whether its
**ranking** is useful. Measured on 1024³:

| tiling | model GB | A4 working set | ms | TFLOP/s |
|---|---|---|---|---|
| 32x32 k=32 | 0.273 | 12 KB | 0.256 | 8.40 |
| 32x256 k=32 | 0.155 | 68 KB | 0.238 | 9.03 |
| 256x32 k=32 | 0.155 | 68 KB | 0.221 | 9.70 |
| 64x64 k=64 | 0.138 | 48 KB | 0.218 | 9.87 |
| 64x64 k=32 | 0.138 | 32 KB | 0.189 | 11.36 |
| 64x128 k=32 | 0.105 | 56 KB | 0.186 | 11.53 |
| **128x64 k=32** | 0.105 | 56 KB | **0.177** | **12.13** |
| 128x128 k=32 | 0.071 | 96 KB | 0.211 | 10.17 |
| 128x128 k=64 | 0.071 | 128 KB | — | `OutOfResources` |

**Spearman rank correlation: +0.810.** The model is a useful ranker
and not a correct one.

**Where it is right.** Both agree the worst configurations are the
small and the thin ones, and that there is a plateau of good square-ish
tiles in the middle. A4's three qualitative gates — tiling beats the
naive loop, square beats thin, k innermost beats k outermost — all
survive contact with hardware.

**Where it is wrong.** The model's top pick is `128x128 k=32`; the
clock's is `128x64 k=32`. Taking the model's choice costs **1.19x**
the best measured time. The model ranks `128x128 k=32` best because it
minimises traffic, and the clock puts it *fourth* — because at 96 KB of
the 99 KB available, only one block fits per SM and occupancy
collapses. The model has no occupancy term, so this is invisible to it.

**Where it is right for the wrong reason.** A4's `working_set` counts
the A tile, the B tile *and* the C accumulator, and ignores
pipelining. The real shared-memory requirement is (A + B) x
`num_stages`, with C living in registers. At `128x128 k=32` A4 reads
96 KB where the kernel uses 64 KB; at `128x128 k=64` both come to
128 KB — by coincidence, two errors cancelling. So the feasibility
filter is roughly right in magnitude and wrong in mechanism.

A4's default budget of 128 KB also exceeds this GPU's 99 KB. Re-running
its search at the measured limit narrows the space from 344 feasible
schedules to 324 and does not change its pick, which is itself
informative: the constraint was not what was binding.

**None of this was changed in A4.** The model is a teaching model, its
gates are honest, and its docstring already says it is crude. What it
lacked was a number for *how* crude, which is now +0.810 and 1.19x.

---

## Reproducing

With a GPU and Triton available, nothing special is needed:

```bash
cd compilers_and_triton
python run_all.py
python notebooks/validate_notebooks.py
```

Both report the GPU in their banners and verify the kernels. Without
a GPU both still pass, with the three kernel stages reported as
`UNVERIFIED` and their references checked — confirmed unchanged after
all eight fixes above.

On Windows, `gpu_setup.py --run` supplies the environment:

```bash
python gpu_setup.py --install
python gpu_setup.py --run run_all.py
python gpu_setup.py --run notebooks/validate_notebooks.py
```

Measured here after packaging: **8 verified** through `--run`, 8
notebooks executing end to end, and **5 verified / 3 unverified** from
plain `python run_all.py` in the same environment.
