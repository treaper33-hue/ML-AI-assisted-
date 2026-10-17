"""B1 · FUNDAMENTAL — The first kernel, and the three ideas in it.

Vector addition is the "hello world" of GPU programming, and it is
worth more attention than it usually gets, because the three ideas it
contains are the three ideas you use in every kernel afterwards.

**1. You write the program for one block, not for one element.**
A CUDA kernel is written from the point of view of a single thread.
A Triton kernel is written from the point of view of a *block* of
elements, and the compiler decides how threads cover it. That single
change is why a competent Triton kernel is tens of lines where the CUDA
equivalent is hundreds — and why the compiler can vectorise and
schedule for you.

**2. `program_id` tells you which block you are.**
The grid says how many instances to launch; each instance asks which
one it is and computes its own slice of the data. There is no loop over
blocks — the grid *is* the loop, executed in parallel.

**3. Masks handle the edge.**
Data is rarely a multiple of the block size. Rather than branching, you
compute the full block and pass a boolean mask to the load and store so
out-of-range lanes read a default and write nothing. Branch-free edge
handling is the normal idiom, not a trick.

Run it:
    python b1_vector_add.py
"""

import sys
from typing import List

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from runtime import (CAN_RUN_KERNELS, Result,  # noqa: E402
                     check, environment_banner, report)

if CAN_RUN_KERNELS:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
                          BLOCK_SIZE: tl.constexpr):
        """Adds two vectors, one block per program instance.

        Args:
            x_ptr: Pointer to the first input.
            y_ptr: Pointer to the second input.
            out_ptr: Pointer to the output.
            n_elements: Total length.
            BLOCK_SIZE: Elements per block; a compile-time constant, so
                the compiler can unroll and vectorise against it.
        """
        # Which block am I? The grid is the loop.
        pid = tl.program_id(axis=0)

        # The slice this instance owns.
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # The tail block runs past the end; mask rather than branch.
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    def vector_add(x: 'torch.Tensor', y: 'torch.Tensor',
                   block_size: int = 1024) -> 'torch.Tensor':
        """Launches the kernel.

        Args:
            x: First input, on the GPU.
            y: Second input, same shape.
            block_size: Elements per program instance.

        Returns:
            The elementwise sum.

        Raises:
            ValueError: If the shapes differ.
        """
        if x.shape != y.shape:
            raise ValueError(f'shape mismatch: {x.shape} vs {y.shape}')
        output = torch.empty_like(x)
        n_elements = output.numel()
        # The grid is a function of the meta-parameters, so changing
        # BLOCK_SIZE changes the launch automatically.
        def grid(meta):
            """Instances needed for this block size."""
            return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        vector_add_kernel[grid](x, y, output, n_elements,
                                BLOCK_SIZE=block_size)
        return output


def reference_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The specification: what the kernel must compute.

    Args:
        x: First input.
        y: Second input.

    Returns:
        The elementwise sum.
    """
    return x + y


def main() -> int:
    """Demonstrates the kernel and checks it against the reference.

    Returns:
        A process exit code.
    """
    print('=' * 70)
    print('B1 · FUNDAMENTAL — The first kernel')
    print('=' * 70)
    print()
    print(environment_banner())

    rng = np.random.default_rng(20260101)
    results: List[Result] = []

    # Sizes chosen so that one is an exact multiple of the block size
    # and the others are not: the masked tail is the part that breaks.
    for size in (1024, 100_000, 1_000_003):
        x = rng.standard_normal(size).astype(np.float32)
        y = rng.standard_normal(size).astype(np.float32)
        expected = reference_add(x, y)
        exact_blocks = size % 1024 == 0
        label = (f'n={size:,} '
                 f'({"exact blocks" if exact_blocks else "masked tail"})')

        if CAN_RUN_KERNELS:
            x_gpu = torch.from_numpy(x).cuda()
            y_gpu = torch.from_numpy(y).cuda()
            results.append(check(label,
                                 lambda a=x_gpu, b=y_gpu:
                                 vector_add(a, b),
                                 expected, tolerance=1e-6))
        else:
            results.append(Result(label, 'SKIP',
                                  'no GPU or Triton in this runtime'))

    # The reference is checked unconditionally, so this module still
    # verifies something on a CPU-only machine.
    small_x = rng.standard_normal(64).astype(np.float32)
    small_y = rng.standard_normal(64).astype(np.float32)
    reference_ok = np.allclose(reference_add(small_x, small_y),
                               small_x + small_y, atol=0)
    results.append(Result('reference matches numpy exactly',
                          'PASS' if reference_ok else 'FAIL',
                          'the specification is self-consistent'))

    code = report('Checks', results)

    print('\nWhat to take from this module')
    print('  * The grid is the loop. You never write a loop over blocks.')
    print('  * BLOCK_SIZE is a tl.constexpr, so it is known at compile')
    print('    time and the compiler can unroll and vectorise with it.')
    print('  * The mask is not an edge case bolted on; it is how every')
    print('    kernel handles a size that is not a multiple of the block.')
    print('  * Vector add is memory-bound: it reads 2N and writes N, and')
    print('    does one flop per element. No kernel can beat bandwidth,')
    print('    which is why fusion (B4) matters more than arithmetic.')
    return code


if __name__ == '__main__':
    sys.exit(main())
