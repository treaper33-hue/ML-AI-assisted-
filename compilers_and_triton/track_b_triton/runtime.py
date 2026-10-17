"""Shared runtime for the Triton track.

Triton compiles and runs on NVIDIA and AMD GPUs. Most people learning
it do not have one to hand, and official Triton wheels are Linux-only,
so every module in this track is built the same way:

1. a **reference implementation** in NumPy that states, unambiguously,
   what the kernel is supposed to compute;
2. a **Triton kernel** that computes it fast;
3. a **check** that runs the reference always and the kernel when a GPU
   is present.

Without a GPU the modules still run, still teach, and report the kernel
as `SKIPPED` rather than pretending. A skip is not a pass — the same
rule as the rest of this repository.

The fastest way to get a GPU for this track is Colab: Runtime → Change
runtime type → T4 GPU. Triton ships with PyTorch there, so nothing
needs installing.
"""

import dataclasses
import time
from typing import Callable, List, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:                         # pragma: no cover
    HAS_TORCH = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:                         # pragma: no cover
    HAS_TRITON = False
    triton = None
    tl = None

HAS_GPU = bool(HAS_TORCH and torch.cuda.is_available())
CAN_RUN_KERNELS = bool(HAS_TRITON and HAS_GPU)

PASS = 'PASS'
FAIL = 'FAIL'
SKIP = 'SKIP'


def environment_banner() -> str:
    """Describes what this machine can actually verify.

    Returns:
        A multi-line description.
    """
    lines = ['environment:']
    if HAS_TORCH:
        lines.append(f'  torch    {torch.__version__}')
        if HAS_GPU:
            major, minor = torch.cuda.get_device_capability(0)
            lines.append(f'  gpu      '
                         f'{torch.cuda.get_device_name(0)} '
                         f'(sm_{major}{minor})')
        else:
            lines.append('  gpu      none visible')
    else:
        lines.append('  torch    not installed')
    lines.append(f'  triton   {"installed" if HAS_TRITON else "not installed"}')
    if CAN_RUN_KERNELS:
        lines.append('  -> kernels will RUN and be checked against the '
                     'reference')
    else:
        lines.append('  -> kernels will be SKIPPED; the reference still '
                     'runs and is checked')
        lines.append('     (Colab: Runtime -> Change runtime type -> T4 GPU)')
    return '\n'.join(lines)


@dataclasses.dataclass
class Result:
    """The outcome of one checked kernel.

    Attributes:
        name: What was checked.
        status: PASS, FAIL or SKIP.
        detail: Numbers a reader can judge.
        milliseconds: Kernel time, when it ran.
    """

    name: str
    status: str
    detail: str = ''
    milliseconds: Optional[float] = None

    def line(self) -> str:
        """Renders one report line."""
        timing = (f'  {self.milliseconds:7.3f} ms'
                  if self.milliseconds is not None else '')
        return f'  [{self.status}] {self.name:<40}{timing}  {self.detail}'


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Returns the largest relative elementwise difference.

    Absolute difference is the wrong measure for anything in float32:
    an error of 1e-3 is catastrophic near zero and invisible near 1e6.

    Args:
        actual: What the kernel produced.
        expected: What the reference produced.

    Returns:
        The maximum relative error.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = np.maximum(np.abs(expected), 1e-6)
    return float(np.max(np.abs(actual - expected) / scale))


def normwise_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Returns ||actual - expected|| / ||expected||, Frobenius norm.

    `relative_error` above divides elementwise, which is the right
    measure when every expected value is bounded away from zero -- a
    softmax probability, a positive activation. It is the wrong measure
    the moment the output straddles zero, because it is then APE, and
    APE is unbounded at zeros: an absolute error of 1e-4 against an
    entry that happens to be 1e-6 reads as 1e+02 and says nothing about
    the computation.

    A matrix product of random data straddles zero everywhere. This is
    not a hypothetical: the elementwise measure reports 6.2e-02 for
    `torch.matmul` on one of B3's own test shapes, so a gate built on
    it rejects the reference implementation of the operation. Numerical
    linear algebra uses the normwise measure for exactly this reason,
    and so does B3.

    Args:
        actual: What the kernel produced.
        expected: What the reference produced.

    Returns:
        The normwise relative error.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = float(np.linalg.norm(expected.reshape(-1)))
    return float(np.linalg.norm((actual - expected).reshape(-1))
                 / max(scale, 1e-300))


def check(name: str, kernel: Callable[[], 'np.ndarray'],
          expected: np.ndarray, tolerance: float = 2e-2,
          repeats: int = 20, metric: str = 'elementwise') -> Result:
    """Runs a kernel, times it, and compares it with the reference.

    Args:
        name: What is being checked.
        kernel: Zero-argument callable returning the kernel's output.
        expected: The reference result.
        tolerance: Maximum acceptable relative error. The default is
            loose because float32 accumulation over a reduction is
            genuinely imprecise; each module tightens it where it can.
        repeats: Timed iterations after a warm-up.
        metric: 'elementwise' (the default) or 'normwise'. Choosing
            this per operation rather than once for the whole suite is
            deliberate -- see `normwise_error` for which is right when.

    Returns:
        The result record.

    Raises:
        ValueError: On an unknown metric name.
    """
    if metric not in ('elementwise', 'normwise'):
        raise ValueError(f'unknown metric {metric!r}')
    if not CAN_RUN_KERNELS:
        return Result(name, SKIP, 'no GPU or Triton in this runtime')
    try:
        output = kernel()                   # warm-up: compiles the kernel
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            output = kernel()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0 / repeats

        array = (output.detach().float().cpu().numpy()
                 if HAS_TORCH and isinstance(output, torch.Tensor)
                 else np.asarray(output))
        if metric == 'normwise':
            error = normwise_error(array, expected)
            description = f'normwise relative error {error:.2e}'
        else:
            error = relative_error(array, expected)
            description = f'max relative error {error:.2e}'
        status = PASS if error < tolerance else FAIL
        return Result(name, status, description, elapsed)
    except Exception as error:              # noqa: BLE001 - reported
        return Result(name, FAIL, f'{type(error).__name__}: {error}')


def report(title: str, results: List[Result]) -> int:
    """Prints a result block and returns a process exit code.

    Args:
        title: Section heading.
        results: What to report.

    Returns:
        0 unless something failed. A skip is not a failure, but it is
        also not a pass, and the summary says so.
    """
    print(f'\n{title}')
    for result in results:
        print(result.line())
    failed = sum(r.status == FAIL for r in results)
    skipped = sum(r.status == SKIP for r in results)
    passed = sum(r.status == PASS for r in results)
    print(f'\n  passed {passed}   failed {failed}   skipped {skipped}')
    if skipped and not failed:
        print('  status: UNVERIFIED — the kernels did not run here')
    elif failed:
        print('  status: FAILED')
    else:
        print('  status: VERIFIED')
    return 1 if failed else 0
