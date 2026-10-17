"""Runs every stage of the compilers-and-Triton ladder in order.

Each stage is a standalone script that prints its own gates. This runs
them in sequence and summarises, the same convention as
`projects/run_all.py` elsewhere in this repository.

Exit codes:
    0  every stage that could run, passed
    1  a stage failed
    2  nothing ran

Usage:
    python run_all.py
    python run_all.py --track a      only the compiler track
    python run_all.py --verbose      show each stage's full output
"""

import argparse
import pathlib
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent

STAGES: List[Tuple[str, str, str, str]] = [
    ('a', 'A1', 'track_a_compilers/a1_lexer_parser.py',
     'FUNDAMENTAL  text to tree: lexing and Pratt parsing'),
    ('a', 'A2', 'track_a_compilers/a2_ir.py',
     'FUNDAMENTAL  tree to three-address code, and an interpreter'),
    ('a', 'A3', 'track_a_compilers/a3_passes.py',
     'INTERMEDIATE folding, copy propagation, CSE, DCE to a fixpoint'),
    ('a', 'A4', 'track_a_compilers/a4_schedule.py',
     'ADVANCED     algorithm vs schedule, cost model, autotuning'),
    ('b', 'B1', 'track_b_triton/b1_vector_add.py',
     'FUNDAMENTAL  program_id, blocks, masks'),
    ('b', 'B2', 'track_b_triton/b2_softmax.py',
     'INTERMEDIATE reductions, and numerically stable softmax'),
    ('b', 'B3', 'track_b_triton/b3_matmul.py',
     'ADVANCED     tiling, register accumulation, grouped ordering'),
    ('c', 'CAP', 'capstone/tensor_compiler.py',
     'CAPSTONE     a tensor compiler that emits fused Triton'),
]


def run_stage(script: str, verbose: bool) -> Dict[str, object]:
    """Runs one stage and captures its verdict.

    Args:
        script: Path relative to this directory.
        verbose: Whether to echo the stage's output.

    Returns:
        A record with the exit code, duration and status line.
    """
    started = time.perf_counter()
    result = subprocess.run([sys.executable, str(HERE / script)],
                            capture_output=True, text=True,
                            encoding='utf-8', errors='replace')
    elapsed = time.perf_counter() - started
    if verbose:
        print(result.stdout)

    status = 'UNKNOWN'
    for line in reversed(result.stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith('status:'):
            # Take the verdict only: modules append an explanation
            # after it, e.g. 'UNVERIFIED - the kernels did not run'.
            status = stripped.split(':', 1)[1].strip().split()[0]
            break
    return {'returncode': result.returncode, 'seconds': elapsed,
            'status': status,
            'stderr': result.stderr.strip().splitlines()[-1:]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument vector.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', choices=('a', 'b', 'c'),
                        help='run one track only')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)

    stages = [s for s in STAGES if not args.track or s[0] == args.track]
    if not stages:
        print('nothing selected')
        return 2

    print('=' * 74)
    print('COMPILERS AND TRITON — every stage')
    print('=' * 74)

    records: List[Tuple[str, str, Dict[str, object]]] = []
    for _, tag, script, description in stages:
        print(f'\n-> {tag}  {description}')
        record = run_stage(script, args.verbose)
        records.append((tag, description, record))
        marker = {'VERIFIED': 'OK ', 'UNVERIFIED': '.. ',
                  'FAILED': '!! '}.get(str(record['status']), '?? ')
        print(f'   {marker}{record["status"]:<12} '
              f'{record["seconds"]:.1f}s')
        if record['returncode'] != 0 and record['status'] != 'UNVERIFIED':
            for line in record['stderr']:
                print(f'       {line}')

    print('\n' + '=' * 74)
    verified = sum(1 for _, _, r in records if r['status'] == 'VERIFIED')
    unverified = sum(1 for _, _, r in records
                     if r['status'] == 'UNVERIFIED')
    failed = sum(1 for _, _, r in records if r['status'] == 'FAILED')
    total_time = sum(float(r['seconds']) for _, _, r in records)

    print(f'{verified} verified   {unverified} unverified   '
          f'{failed} failed   ({total_time:.1f}s)')
    if unverified:
        print('\nUNVERIFIED means the GPU kernels did not run in this')
        print('runtime. The reference implementations were still checked.')
        print('For the kernels: Colab -> Runtime -> Change runtime type')
        print('-> T4 GPU, then open the notebooks in notebooks/.')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
