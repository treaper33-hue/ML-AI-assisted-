"""A4 · ADVANCED — Scheduling: the decisions after correctness.

The passes in A3 changed *what* is computed. Scheduling changes *how
and in what order*, without changing the result at all. For tensor
programs this is where essentially all the performance lives, because
the arithmetic is fixed and the memory traffic is not.

The separation this module teaches — **algorithm versus schedule** — is
the organising idea of Halide, TVM and every tensor compiler since.
The algorithm says `C[i,j] = sum_k A[i,k] * B[k,j]`. The schedule says
in what order, in what tile sizes, and with what held in registers. The
same algorithm with two schedules can differ by an order of magnitude.

A cost model stands in for the hardware here, because a model you can
read beats a benchmark you cannot explain. It counts memory traffic
under a simple cache, which is crude and still ranks tilings correctly
— and the gates check that it reproduces the three facts every
performance engineer knows about matrix multiply.

Run it:
    python a4_schedule.py
"""

import dataclasses
import itertools
import sys
from typing import Iterator, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class Schedule:
    """One way of executing a matrix multiply.

    Attributes:
        tile_m: Rows of C computed per tile.
        tile_n: Columns of C computed per tile.
        tile_k: Depth processed per step.
        order: Loop nest order, a permutation of 'mnk'.
    """

    tile_m: int
    tile_n: int
    tile_k: int
    order: str

    def __str__(self) -> str:
        """Renders the schedule compactly."""
        return (f'{self.tile_m:>3}x{self.tile_n:<3} k={self.tile_k:<3} '
                f'order={self.order}')


def working_set(schedule: Schedule, element_bytes: int = 4) -> int:
    """Bytes a single tile needs resident at once.

    A tile of A is tile_m by tile_k, a tile of B is tile_k by tile_n,
    and the accumulator for C is tile_m by tile_n. If all three fit in
    fast memory the inner loop never touches DRAM.

    Args:
        schedule: The schedule to measure.
        element_bytes: Size of one element.

    Returns:
        The working set in bytes.
    """
    a = schedule.tile_m * schedule.tile_k
    b = schedule.tile_k * schedule.tile_n
    c = schedule.tile_m * schedule.tile_n
    return (a + b + c) * element_bytes


def estimate_traffic(schedule: Schedule, m: int, n: int, k: int,
                     cache_bytes: int = 128 * 1024,
                     element_bytes: int = 4) -> float:
    """Estimates DRAM traffic in bytes for one matrix multiply.

    The model: a tile whose working set fits in cache reads each of its
    operands once; a tile that does not fit re-reads them from DRAM on
    every pass. That single threshold is enough to reproduce why
    tiling helps and why tiles that are too large stop helping.

    Args:
        schedule: The schedule.
        m: Rows of A and C.
        n: Columns of B and C.
        k: Shared dimension.
        cache_bytes: Fast-memory budget.
        element_bytes: Size of one element.

    Returns:
        Estimated bytes moved; lower is better.
    """
    tiles_m = max(1, m // schedule.tile_m)
    tiles_n = max(1, n // schedule.tile_n)
    tiles_k = max(1, k // schedule.tile_k)

    a_tile = schedule.tile_m * schedule.tile_k * element_bytes
    b_tile = schedule.tile_k * schedule.tile_n * element_bytes
    c_tile = schedule.tile_m * schedule.tile_n * element_bytes

    fits = working_set(schedule, element_bytes) <= cache_bytes
    penalty = 1.0 if fits else 4.0

    # Each (m, n) tile walks the whole k dimension, reading a strip of
    # A and a strip of B per step.
    reads = tiles_m * tiles_n * tiles_k * (a_tile + b_tile) * penalty

    # Where the k loop is outermost, C is re-read and re-written per
    # step instead of being accumulated in registers.
    if schedule.order.index('k') == 0:
        writes = tiles_m * tiles_n * tiles_k * c_tile * 2
    else:
        writes = tiles_m * tiles_n * c_tile

    return reads + writes


def naive_traffic(m: int, n: int, k: int, element_bytes: int = 4) -> float:
    """Traffic for the textbook triple loop with no tiling.

    Every element of C re-reads a full row of A and a full column of B.

    Args:
        m: Rows.
        n: Columns.
        k: Shared dimension.
        element_bytes: Size of one element.

    Returns:
        Estimated bytes moved.
    """
    return float(m * n * k * 2 * element_bytes + m * n * element_bytes)


def search_space(cache_bytes: int = 128 * 1024) -> Iterator[Schedule]:
    """Enumerates candidate schedules.

    Only schedules whose working set fits the budget are produced:
    searching configurations that cannot run wastes the budget, and
    this is exactly what `triton.autotune` does with its config list.

    Args:
        cache_bytes: Fast-memory budget.

    Yields:
        Feasible schedules.
    """
    sizes = (16, 32, 64, 128, 256)
    depths = (8, 16, 32, 64)
    orders = ('mnk', 'nmk', 'mkn', 'kmn')
    for tile_m, tile_n, tile_k, order in itertools.product(
            sizes, sizes, depths, orders):
        candidate = Schedule(tile_m, tile_n, tile_k, order)
        if working_set(candidate) <= cache_bytes:
            yield candidate


def autotune(m: int, n: int, k: int,
             cache_bytes: int = 128 * 1024) -> Tuple[Schedule, float, int]:
    """Searches the space for the cheapest schedule.

    Exhaustive here because the space is small. Real autotuners sample
    it, or model it, or both — but the structure is identical: generate
    feasible configurations, score them, keep the best.

    Args:
        m: Rows.
        n: Columns.
        k: Shared dimension.
        cache_bytes: Fast-memory budget.

    Returns:
        A tuple (best schedule, its traffic, configurations tried).
    """
    best: Optional[Schedule] = None
    best_cost = float('inf')
    tried = 0
    for candidate in search_space(cache_bytes):
        tried += 1
        cost = estimate_traffic(candidate, m, n, k, cache_bytes)
        if cost < best_cost:
            best, best_cost = candidate, cost
    assert best is not None
    return best, best_cost, tried


def main() -> int:
    """Runs the search and checks the model's qualitative claims.

    Returns:
        0 when every gate passed.
    """
    print('=' * 70)
    print('A4 · ADVANCED — Scheduling: the decisions after correctness')
    print('=' * 70)

    m = n = k = 1024
    print(f'\nproblem: C[{m}x{n}] = A[{m}x{k}] @ B[{k}x{n}], float32')

    baseline = naive_traffic(m, n, k)
    best, cost, tried = autotune(m, n, k)

    print(f'\nsearched {tried} feasible schedules')
    print(f'  naive triple loop   {baseline / 1e9:8.2f} GB')
    print(f'  best schedule       {cost / 1e9:8.2f} GB   ({best})')
    print(f'  improvement         {baseline / cost:8.1f}x less traffic')
    print('\nThe arithmetic is identical in both. Only the order and the')
    print('tile sizes changed, and that is the whole performance story.')

    print('\nTop five schedules')
    ranked = sorted(((estimate_traffic(s, m, n, k), s)
                     for s in search_space()), key=lambda pair: pair[0])
    for rank, (score, schedule) in enumerate(ranked[:5], 1):
        print(f'  {rank}. {schedule}   {score / 1e9:7.3f} GB   '
              f'working set {working_set(schedule) / 1024:6.1f} KB')

    failures = 0
    print('\nGates — the model reproduces what is actually true')

    ok = cost < baseline
    failures += not ok
    print(f'  [{"PASS" if ok else "FAIL"}] tiling beats the naive loop '
          f'({baseline / cost:.1f}x)')

    square = estimate_traffic(Schedule(64, 64, 32, 'mnk'), m, n, k)
    thin = estimate_traffic(Schedule(16, 256, 32, 'mnk'), m, n, k)
    ok = square < thin
    failures += not ok
    print(f'  [{"PASS" if ok else "FAIL"}] square tiles beat thin ones '
          f'({square / 1e9:.2f} vs {thin / 1e9:.2f} GB) — a square tile '
          f'maximises reuse per byte loaded')

    accumulate = estimate_traffic(Schedule(64, 64, 32, 'mnk'), m, n, k)
    k_outer = estimate_traffic(Schedule(64, 64, 32, 'kmn'), m, n, k)
    ok = accumulate < k_outer
    failures += not ok
    print(f'  [{"PASS" if ok else "FAIL"}] k innermost beats k outermost '
          f'({accumulate / 1e9:.2f} vs {k_outer / 1e9:.2f} GB) — k outside '
          f'forces C in and out of memory every step')

    oversized = Schedule(256, 256, 64, 'mnk')
    ok = working_set(oversized) > 128 * 1024
    failures += not ok
    print(f'  [{"PASS" if ok else "FAIL"}] a 256x256 tile with k=64 '
          f'({working_set(oversized) / 1024:.0f} KB) is excluded from the '
          f'search: it does not fit')

    every_feasible = all(working_set(s) <= 128 * 1024
                         for s in search_space())
    failures += not every_feasible
    print(f'  [{"PASS" if every_feasible else "FAIL"}] every candidate '
          f'generated is feasible')

    print(f'\nstatus: {"VERIFIED" if not failures else "FAILED"}')
    print('\nWhere this goes next')
    print('  * B3 writes the tiled kernel this schedule describes.')
    print('  * triton.autotune does this search against the real clock')
    print('    instead of a model, which is better and far slower.')
    print('  * The capstone emits a kernel; a production compiler emits')
    print('    a kernel *and* chooses its schedule, which is this.')
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
