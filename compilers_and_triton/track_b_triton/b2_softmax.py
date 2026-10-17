"""B2 · INTERMEDIATE — Reductions, and why softmax is written oddly.

Vector add was *elementwise*: every output depends on one input, so
blocks never talk to each other. A reduction is the first thing that
breaks that. Softmax over a row needs the maximum of the row and the
sum of the row before it can produce any output at all.

Two lessons, and the second is the one that bites people.

**1. Give each program instance a whole row.**
If a row fits in a block, the entire reduction happens in registers and
no cross-block communication is needed. One program per row, a block
wide enough for the row, and `tl.max` / `tl.sum` do the rest. When rows
do not fit you need a two-pass algorithm or an online one; this module
takes the common case and says so.

**2. Subtract the maximum, or overflow.**
exp(x) overflows float32 above about 88. Logits of 100 are ordinary.
The identity

    softmax(x) = softmax(x - c)   for any constant c

holds exactly, because the constant cancels between numerator and
denominator. Taking c = max(x) makes the largest exponent exp(0) = 1
and everything else smaller, so nothing overflows. That is why every
softmax implementation you will ever read subtracts the max first, and
it is not a numerical nicety — the naive version returns NaN on real
data.

Run it:
    python b2_softmax.py
"""

import sys
from typing import List

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from runtime import (CAN_RUN_KERNELS, Result, check,  # noqa: E402
                     environment_banner, relative_error, report)

if CAN_RUN_KERNELS:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                       n_cols, BLOCK_SIZE: tl.constexpr):
        """Computes a numerically stable softmax, one row per program.

        Args:
            out_ptr: Output pointer.
            in_ptr: Input pointer.
            in_row_stride: Elements between consecutive input rows.
            out_row_stride: Elements between consecutive output rows.
            n_cols: Row length.
            BLOCK_SIZE: Padded row width; a power of two >= n_cols.
        """
        row = tl.program_id(axis=0)
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < n_cols

        # Masked lanes load -inf so they lose the max and contribute
        # exp(-inf) = 0 to the sum. The mask does the padding for free.
        values = tl.load(in_ptr + row * in_row_stride + columns,
                         mask=mask, other=-float('inf'))

        shifted = values - tl.max(values, axis=0)
        numerator = tl.exp(shifted)
        denominator = tl.sum(numerator, axis=0)

        tl.store(out_ptr + row * out_row_stride + columns,
                 numerator / denominator, mask=mask)

    def softmax(x: 'torch.Tensor') -> 'torch.Tensor':
        """Launches one program instance per row.

        Args:
            x: A 2-D tensor on the GPU.

        Returns:
            Row-wise softmax.

        Raises:
            ValueError: If a row is too wide for one block.
        """
        n_rows, n_cols = x.shape
        block_size = triton.next_power_of_2(n_cols)
        if block_size > 65536:
            raise ValueError(
                f'rows of {n_cols} do not fit one block; this kernel '
                f'assumes they do, and a wider row needs the two-pass '
                f'or online algorithm')
        output = torch.empty_like(x)
        softmax_kernel[(n_rows,)](
            output, x, x.stride(0), output.stride(0), n_cols,
            BLOCK_SIZE=block_size,
            num_warps=min(max(block_size // 256, 1), 16))
        return output


def reference_softmax(x: np.ndarray) -> np.ndarray:
    """The specification: stable row-wise softmax.

    Args:
        x: A 2-D array.

    Returns:
        Row-wise softmax, computed in float64.
    """
    x = np.asarray(x, dtype=np.float64)
    shifted = x - x.max(axis=1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=1, keepdims=True)


def naive_softmax(x: np.ndarray) -> np.ndarray:
    """The version everybody writes first, in float32, without the shift.

    Kept so the failure can be demonstrated rather than asserted.

    Args:
        x: A 2-D array.

    Returns:
        Row-wise softmax, or NaN where exp overflowed.
    """
    x = np.asarray(x, dtype=np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        numerator = np.exp(x)
        return numerator / numerator.sum(axis=1, keepdims=True)


def main() -> int:
    """Demonstrates the numerics lesson and checks the kernel.

    Returns:
        A process exit code.
    """
    print('=' * 70)
    print('B2 · INTERMEDIATE — Reductions, and stable softmax')
    print('=' * 70)
    print()
    print(environment_banner())

    print('\n' + '-' * 70)
    print('Why the maximum is subtracted')
    print('-' * 70)
    logits = np.array([[100.0, 101.0, 102.0]], dtype=np.float32)
    naive = naive_softmax(logits)
    stable = reference_softmax(logits)
    print(f'\n  logits               {logits[0]}')
    print(f'  naive   exp(x)/sum   {naive[0]}')
    print(f'  stable  exp(x-max)   {np.round(stable[0], 6)}')
    print('\n  exp(102) = inf in float32, so the naive version divides')
    print('  inf by inf and returns NaN. These are not exotic logits.')

    failures = 0
    naive_broke = bool(np.any(~np.isfinite(naive)))
    stable_fine = bool(np.all(np.isfinite(stable)))
    failures += not (naive_broke and stable_fine)
    print(f'\n  [{"PASS" if naive_broke else "FAIL"}] the naive version '
          f'does produce NaN here')
    print(f'  [{"PASS" if stable_fine else "FAIL"}] the stable version '
          f'does not')

    rows = reference_softmax(np.array([[100.0, 101.0, 102.0],
                                       [-50.0, 0.0, 50.0]]))
    sums_ok = np.allclose(rows.sum(axis=1), 1.0, atol=1e-12)
    failures += not sums_ok
    print(f'  [{"PASS" if sums_ok else "FAIL"}] every row sums to 1 '
          f'(max row error {abs(rows.sum(axis=1) - 1).max():.1e})')

    shift = reference_softmax(np.array([[1.0, 2.0, 3.0]]))
    shifted = reference_softmax(np.array([[1.0, 2.0, 3.0]]) + 1000.0)
    invariant = relative_error(shifted, shift) < 1e-12
    failures += not invariant
    print(f'  [{"PASS" if invariant else "FAIL"}] softmax(x) == '
          f'softmax(x + 1000), as the identity requires')

    rng = np.random.default_rng(20260101)
    results: List[Result] = []
    for n_rows, n_cols in ((128, 256), (1024, 781), (64, 2048)):
        x = (rng.standard_normal((n_rows, n_cols)) * 10).astype(np.float32)
        expected = reference_softmax(x)
        label = f'softmax {n_rows}x{n_cols}'
        if CAN_RUN_KERNELS:
            x_gpu = torch.from_numpy(x).cuda()
            results.append(check(label,
                                 lambda a=x_gpu: softmax(a),
                                 expected,
                                 tolerance=1e-5))
        else:
            results.append(Result(label, 'SKIP',
                                  'no GPU or Triton in this runtime'))

    code = report('Kernel checks', results)

    print('\nWhat to take from this module')
    print('  * A reduction forces a decision the elementwise case did')
    print('    not: who owns the data being reduced. One row per program')
    print('    keeps the whole reduction in registers.')
    print('  * Masked lanes load -inf, not 0. Loading 0 would make an')
    print('    all-negative row return the wrong maximum.')
    print('  * num_warps is tuned to the block width. Too few and the')
    print('    reduction serialises; too many and most sit idle.')
    print('  * The stability trick is exact, not approximate: the')
    print('    constant cancels algebraically.')
    return code if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
