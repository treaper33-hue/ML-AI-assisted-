"""Runs all four worked projects and summarises their gates.

Each project is a standalone script; this just runs them in order and
collects the verdicts, so one command tells you whether the whole set
still works on this machine.

Usage:
    python run_all.py
    python run_all.py --quick       smaller budgets where supported
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

HERE = pathlib.Path(__file__).resolve().parent

PROJECTS = [
    ('p1_tabular_churn.py', 'Tabular classification, end to end',
     ['--quick']),
    ('p2_digits_cnn.py', 'A convolutional classifier, trained by hand',
     ['--epochs', '15']),
    ('p3_forecast_conformal.py', 'Forecasting with real intervals', []),
    ('p4_vae_elbo.py', 'A VAE with the ELBO taken apart',
     ['--epochs', '30']),
    ('p5_serving_monitoring.py', 'Serving a model, and watching it after',
     ['--requests', '200']),
]


def run_one(script: str, extra: Sequence[str]) -> Dict[str, object]:
    """Runs one project and returns its verdict.

    Args:
        script: File name of the project.
        extra: Extra command-line arguments.

    Returns:
        A dict with the exit code, duration and parsed report, if any.
    """
    started = time.perf_counter()
    result = subprocess.run([sys.executable, str(HERE / script), *extra],
                            capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    status, gates = 'UNKNOWN', []
    name = script.replace('.py', '')
    report_path = HERE / 'runs' / name / 'report.json'
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding='utf-8'))
        status = str(payload.get('status', 'UNKNOWN'))
        gates = payload.get('gates', [])
    elif result.returncode == 2:
        status = 'UNVERIFIED (missing dependency)'

    return {
        'script': script,
        'returncode': result.returncode,
        'seconds': round(elapsed, 1),
        'status': status,
        'gates_passed': sum(1 for g in gates if g.get('passed')),
        'gates_total': len(gates),
        'tail': result.stdout.strip().splitlines()[-1:] if result.stdout
                else [],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument vector.

    Returns:
        0 when every project verified.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true',
                        help='smaller budgets where supported')
    args = parser.parse_args(argv)

    print('=' * 74)
    print('ALL PROJECTS')
    print('=' * 74)

    results: List[Dict[str, object]] = []
    for script, title, extra in PROJECTS:
        if args.quick and script == 'p2_digits_cnn.py':
            extra = ['--epochs', '8', '--skip-lr-test']
        if args.quick and script == 'p4_vae_elbo.py':
            extra = ['--epochs', '15', '--skip-sweep']
        if args.quick and script == 'p5_serving_monitoring.py':
            extra = ['--requests', '60']
        print(f'\n-> {script}: {title}')
        record = run_one(script, extra)
        results.append(record)
        mark = 'OK ' if record['status'] == 'VERIFIED' else '!! '
        print(f'   {mark}{record["status"]}  '
              f'gates {record["gates_passed"]}/{record["gates_total"]}  '
              f'{record["seconds"]}s')

    print('\n' + '=' * 74)
    verified = sum(1 for r in results if r['status'] == 'VERIFIED')
    total_time = sum(float(r['seconds']) for r in results)
    print(f'{verified}/{len(results)} projects VERIFIED in {total_time:.0f}s')
    for record in results:
        if record['status'] != 'VERIFIED':
            print(f'  {record["script"]}: {record["status"]} '
                  f'(exit {record["returncode"]})')
    return 0 if verified == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
