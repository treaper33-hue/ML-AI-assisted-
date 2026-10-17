"""B3 · ADVANCED — The tiled matrix multiply A4 described.

A4 searched schedules and concluded: square tiles, k innermost,
accumulate in registers. This module writes that kernel.

The shape of it:

* each program instance owns one **tile of C**, say 128x128;
* it walks the k dimension in steps of BLOCK_K, loading a strip of A
  and a strip of B each step;
* it accumulates into a register block and stores **once**, at the end.

That last point is the whole game. The naive loop writes every partial
sum back to memory; this one keeps the accumulator in registers for the
entire k loop, so C is written exactly once. A4's cost model predicted
the difference; this is the code that realises it.

**Grouped ordering.** Program instances are numbered linearly, and the
obvious row-major mapping means consecutive instances share a row of A
but stride across all of B, so the L2 cache sees no reuse. Grouping
them into blocks of `GROUP_M` rows makes neighbouring instances share
both operands. It is a permutation of the launch order and nothing
else — and it is routinely worth tens of percent.

Run it:
    python b3_matmul.py
"""

import sys
import time
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from runtime import (CAN_RUN_KERNELS, Result, check,  # noqa: E402
                     environment_banner, normwise_error, report)

if CAN_RUN_KERNELS:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                      PRECISION: tl.constexpr):
        """Computes one BLOCK_M x BLOCK_N tile of C = A @ B.

        Args:
            a_ptr: Pointer to A.
            b_ptr: Pointer to B.
            c_ptr: Pointer to C.
            M: Rows of A and C.
            N: Columns of B and C.
            K: Shared dimension.
            stride_am: Row stride of A.
            stride_ak: Column stride of A.
            stride_bk: Row stride of B.
            stride_bn: Column stride of B.
            stride_cm: Row stride of C.
            stride_cn: Column stride of C.
            BLOCK_M: Tile rows.
            BLOCK_N: Tile columns.
            BLOCK_K: Depth per step.
            GROUP_M: Tiles per group, for L2 reuse.
            PRECISION: 'ieee' for true float32 multiplication, or
                'tf32' to let the tensor cores round the inputs to a
                10-bit mantissa first. See `precision_comparison`.
        """
        pid = tl.program_id(axis=0)
        tiles_m = tl.cdiv(M, BLOCK_M)
        tiles_n = tl.cdiv(N, BLOCK_N)

        # Grouped ordering: permute the linear program id so that
        # neighbouring instances reuse the same strips of A and B.
        tiles_per_group = GROUP_M * tiles_n
        group = pid // tiles_per_group
        first_m = group * GROUP_M
        group_rows = tl.minimum(tiles_m - first_m, GROUP_M)
        pid_m = first_m + ((pid % tiles_per_group) % group_rows)
        pid_n = (pid % tiles_per_group) // group_rows

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am
                          + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk
                          + offs_n[None, :] * stride_bn)

        # The accumulator lives in registers for the whole k loop and
        # is written to memory exactly once, at the end. float32 even
        # for float16 inputs: accumulating a long sum in half precision
        # loses accuracy badly.
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for step in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k[None, :] < K - step * BLOCK_K
            a = tl.load(a_ptrs, mask=k_mask, other=0.0)
            b = tl.load(b_ptrs, mask=k_mask.T, other=0.0)
            # input_precision is NOT optional for a kernel that
            # claims float32. Left out, tl.dot rounds both inputs to
            # TF32 on any tensor-core GPU -- measurably, 7.7e-04
            # normwise against 5.7e-07 for 'ieee'. The accumulator
            # being float32 does not save it: the rounding happens
            # before the multiply.
            accumulator += tl.dot(a, b, input_precision=PRECISION)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm
                          + offs_cn[None, :] * stride_cn)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    def matmul(a: 'torch.Tensor', b: 'torch.Tensor',
               block_m: int = 128, block_n: int = 128,
               block_k: int = 32, group_m: int = 8,
               precision: str = 'ieee') -> 'torch.Tensor':
        """Launches one program instance per tile of C.

        Args:
            a: Left operand, 2-D, on the GPU.
            b: Right operand, 2-D, on the GPU.
            block_m: Tile rows.
            block_n: Tile columns.
            block_k: Depth per step.
            group_m: Tiles per group.
            precision: 'ieee' or 'tf32'. The default is 'ieee',
                because a function returning float32 should have
                multiplied in float32; 'tf32' is 1.7x faster and
                three orders of magnitude less accurate.

        Returns:
            The product.

        Raises:
            ValueError: If the inner dimensions disagree.
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError(f'cannot multiply {tuple(a.shape)} by '
                             f'{tuple(b.shape)}')
        m, k = a.shape
        _, n = b.shape
        c = torch.empty((m, n), device=a.device, dtype=torch.float32)
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        matmul_kernel[grid](
            a, b, c, m, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP_M=group_m, PRECISION=precision)
        return c


def reference_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The specification, in float64 so it is the trustworthy answer.

    Args:
        a: Left operand.
        b: Right operand.

    Returns:
        The product.
    """
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)


def accumulator_precision_demo() -> float:
    """Shows why the accumulator is float32 even for float16 inputs.

    Returns:
        The relative error of a float16 accumulation of 4096 terms.
    """
    rng = np.random.default_rng(7)
    values = rng.standard_normal(4096).astype(np.float16)
    half_sum = np.float16(0)
    for value in values:
        half_sum = np.float16(half_sum + value)
    exact = float(np.asarray(values, dtype=np.float64).sum())
    return abs(float(half_sum) - exact) / max(abs(exact), 1e-9)


def best_milliseconds(call, rounds: int = 5, repeats: int = 20) -> float:
    """Times a GPU call and returns the fastest round, not the mean.

    The mean is the wrong statistic on a laptop GPU. Clocks move with
    temperature and with whatever else is using the device, and those
    effects only ever make a measurement slower -- so the mean of a
    noisy sample estimates the noise, not the kernel. The minimum is
    the standard choice for the same reason, and it is why an early
    draft of this module reported `torch.matmul` at 0.802 ms in one
    run and 0.177 ms in another.

    Args:
        call: Zero-argument callable launching the work.
        rounds: Independent timing rounds.
        repeats: Launches per round.

    Returns:
        Milliseconds per launch, from the fastest round.
    """
    call()                                  # warm-up: compiles it
    torch.cuda.synchronize()
    best = float('inf')
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(repeats):
            call()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) * 1000.0 / repeats)
    return best


def precision_comparison(size: int = 1024) -> List[Tuple[str, float, float]]:
    """Measures what TF32 costs in accuracy and buys in speed.

    This exists because the first GPU run of this module failed, and
    the reason was not the tiling. `tl.dot` rounds float32 inputs to
    TF32 unless told otherwise, so a kernel that accumulates in
    float32 and returns float32 can still be multiplying with a 10-bit
    mantissa. Nothing in the code said so; only hardware did.

    Args:
        size: Square problem dimension.

    Returns:
        A list of (label, normwise relative error, milliseconds).
    """
    if not CAN_RUN_KERNELS:
        return []

    rng = np.random.default_rng(20260101)
    a = rng.standard_normal((size, size)).astype(np.float32)
    b = rng.standard_normal((size, size)).astype(np.float32)
    expected = reference_matmul(a, b)
    a_gpu = torch.from_numpy(a).cuda()
    b_gpu = torch.from_numpy(b).cuda()

    rows: List[Tuple[str, float, float]] = []
    for label, precision in (('tl.dot ieee', 'ieee'),
                             ('tl.dot tf32', 'tf32')):
        def call(p: str = precision) -> 'torch.Tensor':
            """Runs the kernel at one precision."""
            return matmul(a_gpu, b_gpu, precision=p)

        error = normwise_error(call().cpu().numpy(), expected)
        rows.append((label, error, best_milliseconds(call)))

    def torch_call() -> 'torch.Tensor':
        """Runs torch's own matmul, as a second opinion."""
        return a_gpu @ b_gpu

    rows.append(('torch.matmul',
                 normwise_error(torch_call().cpu().numpy(), expected),
                 best_milliseconds(torch_call)))
    return rows


def main() -> int:
    """Checks the kernel against the reference.

    Returns:
        A process exit code.
    """
    print('=' * 70)
    print('B3 · ADVANCED — The tiled matrix multiply')
    print('=' * 70)
    print()
    print(environment_banner())

    print('\nWhy the accumulator is float32')
    error = accumulator_precision_demo()
    print('  summing 4096 float16 values in float16 gives a relative')
    print(f'  error of {error:.2e}. A k loop of 4096 is ordinary, and')
    print('  that error lands in every element of C. float32')
    print('  accumulation costs nothing on tensor cores and fixes it.')

    rng = np.random.default_rng(20260101)
    results: List[Result] = []
    for m, n, k in ((256, 256, 256), (512, 384, 640), (300, 200, 101)):
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        expected = reference_matmul(a, b)
        ragged = '' if m % 128 == 0 and n % 128 == 0 else ' (ragged tiles)'
        label = f'matmul {m}x{k} @ {k}x{n}{ragged}'
        if CAN_RUN_KERNELS:
            a_gpu = torch.from_numpy(a).cuda()
            b_gpu = torch.from_numpy(b).cuda()
            # metric='normwise', not the elementwise default. C
            # straddles zero, so the elementwise measure is APE and
            # unbounded there -- it scores torch.matmul itself at
            # 6.2e-02 on the middle shape below. runtime.normwise_error
            # has the argument in full.
            results.append(check(label,
                                 lambda a=a_gpu, b=b_gpu:
                                 matmul(a, b),
                                 expected, tolerance=1e-5,
                                 metric='normwise'))
        else:
            results.append(Result(label, 'SKIP',
                                  'no GPU or Triton in this runtime'))

    small_a = rng.standard_normal((8, 5))
    small_b = rng.standard_normal((5, 7))
    reference_ok = np.allclose(reference_matmul(small_a, small_b),
                               small_a @ small_b, atol=1e-12)
    results.append(Result('reference matches numpy',
                          'PASS' if reference_ok else 'FAIL',
                          'the specification is self-consistent'))

    code = report('Kernel checks', results)

    rows = precision_comparison()
    if rows:
        print('\n' + '-' * 70)
        print('What TF32 costs, and what it buys')
        print('-' * 70)
        print(f'\n  {"":<16}{"normwise error":>16}{"ms":>9}{"TFLOP/s":>10}')
        flops = 2.0 * 1024 ** 3
        for label, error, milliseconds in rows:
            print(f'  {label:<16}{error:>16.2e}{milliseconds:>9.3f}'
                  f'{flops / (milliseconds * 1e-3) / 1e12:>10.2f}')

        ieee_error = rows[0][1]
        tf32_error = rows[1][1]
        ieee_ms = rows[0][2]
        tf32_ms = rows[1][2]
        print(f'\n  TF32 is {tf32_error / ieee_error:,.0f}x less accurate '
              f'and {ieee_ms / tf32_ms:.1f}x faster. That is a reasonable')
        print('  trade for training a network and an unreasonable one for')
        print('  solving a linear system. The point is that it is a trade,')
        print('  and leaving input_precision unset makes it silently.')

        print('\nGates on the precision claim')
        tf32_worse = tf32_error > ieee_error * 100
        code = code or 0
        if not tf32_worse:
            code = 1
        print(f'  [{"PASS" if tf32_worse else "FAIL"}] TF32 is measurably '
              f'less accurate ({tf32_error:.1e} vs {ieee_error:.1e}) — '
              f'demonstrated, not asserted')
        ieee_is_float32 = ieee_error < 1e-5
        if not ieee_is_float32:
            code = 1
        print(f'  [{"PASS" if ieee_is_float32 else "FAIL"}] with '
              f"input_precision='ieee' the kernel really is float32 "
              f'({ieee_error:.1e})')

    print('\nWhat to take from this module')
    print('  * The accumulator never leaves registers during the k loop.')
    print('    That is A4\'s "k innermost", realised.')
    print('  * Grouped ordering is a permutation of the launch order and')
    print('    nothing else, and it is worth tens of percent in L2.')
    print('  * tl.dot maps onto tensor cores, and by default it rounds')
    print('    float32 inputs to TF32 to do it. A float32 accumulator')
    print('    does not make a float32 matmul.')
    print('  * Ragged shapes are handled by masks, exactly as in B1.')
    print('  * "Relative error" is not one quantity. Elementwise is')
    print('    wrong wherever the output crosses zero; this module uses')
    print('    the normwise measure and says why.')
    return code


if __name__ == '__main__':
    sys.exit(main())
