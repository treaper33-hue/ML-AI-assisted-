"""Verification entry point for the Auditor role.

Two jobs:

  * `--env` records what this machine actually has, which is what decides
    whether an arm can be verified at all (ledger GAP-13).
  * `python -m mlkit.verify runs/<id>` re-reads a finished run and reports
    its gates and status without trusting anything the Coder said.

Usage:
    python -m mlkit.verify --env
    python -m mlkit.verify runs/20260118T093000-exp01_baseline
"""

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

from mlkit import STATUS_BUG
from mlkit import STATUS_UNVERIFIED
from mlkit import STATUS_VERIFIED
from mlkit import report as report_lib

REQUIRED = ('numpy',)
STAGE_PACKAGES = {
    'scipy': 'stages 1, 4',
    'pandas': 'stages 2-5',
    'sklearn': 'stage 2',
    'torch': 'stages 3-5',
    'matplotlib': 'all stages (plots)',
}

# Import name -> pip name, for the --missing listing. They differ often
# enough that guessing is a reliable way to install the wrong thing.
PIP_NAMES = {
    'sklearn': 'scikit-learn',
    'statsmodels': 'statsmodels',
    'pyarrow': 'pyarrow',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'pandas': 'pandas',
    'torch': 'torch',
    'matplotlib': 'matplotlib',
    'pytest': 'pytest',
    'ruff': 'ruff',
    'mypy': 'mypy',
    'tqdm': 'tqdm',
}


def print_missing() -> int:
    """Prints the pip names of packages this environment lacks.

    Used by the setup scripts when they target an environment that
    already exists: there we install what is absent and leave every
    installed version alone, because downgrading somebody's working
    environment to match our pins is not a service.

    Returns:
        Always 0; an empty line means nothing is missing.
    """
    absent = []
    for import_name, pip_name in sorted(PIP_NAMES.items()):
        try:
            __import__(import_name)
        except ImportError:
            absent.append(pip_name)
        except Exception:  # pylint: disable=broad-except
            # Importable but broken is not missing. Reinstalling over a
            # damaged install is how environments get worse, so we leave
            # it for a human and say nothing here.
            continue
    print(' '.join(absent))
    return 0


def print_environment() -> int:
    """Prints the interpreter and package inventory.

    Returns:
        0 when every required package is present, 1 otherwise.
    """
    info = report_lib.env_info()
    print('environment')
    print(f'  python    : {info["python"]}')
    print(f'  executable: {info["executable"]}')
    print(f'  platform  : {info["platform"]}')
    print('  packages  :')
    missing_required = []
    broken = []
    for name, version in sorted(info['packages'].items()):
        need = STAGE_PACKAGES.get(name, '')
        suffix = f'  <- needed for {need}' if need else ''
        if version == 'MISSING':
            print(f'    {name:<12} MISSING{suffix}')
            if name in REQUIRED:
                missing_required.append(name)
        elif version.startswith('BROKEN'):
            print(f'    {name:<12} {version}{suffix}')
            broken.append(name)
        else:
            print(f'    {name:<12} {version}')
    if missing_required:
        print(
            f'\n  {STATUS_UNVERIFIED}: required package(s) missing: '
            f'{", ".join(missing_required)}'
        )
        return 1
    absent = [k for k, v in info['packages'].items() if v == 'MISSING']
    if absent:
        print(
            '\n  note: arms needing the packages above will report '
            f'{STATUS_UNVERIFIED} (ledger GAP-13); that is the designed '
            'behaviour, not a failure.'
        )
    if broken:
        print(
            f'\n  {STATUS_BUG}: installed but failing to import: '
            f'{", ".join(broken)}. Arms needing these are tagged BUG, '
            'not skipped, and they block the stage gate until the '
            'install is repaired. Log it in VERIFICATION.md.'
        )
    return 0


def verify_run(run_dir: pathlib.Path) -> int:
    """Re-reads a run directory and reports its gates.

    Args:
        run_dir: Directory written by `mlkit.report.write_run`.

    Returns:
        0 when the run is verified, 1 otherwise.

    Raises:
        FileNotFoundError: If the directory has no report.json.
    """
    payload_path = run_dir / 'report.json'
    if not payload_path.exists():
        raise FileNotFoundError(f'no report.json in {run_dir}')
    payload = json.loads(payload_path.read_text(encoding='utf-8'))

    print(f'run        : {run_dir.name}')
    print(f'experiment : {payload["experiment"]}')
    print(f'seed       : {payload["seed"]}')
    card = payload.get('dataset_card', {})
    print(f'dataset    : {card.get("name")} (checksum {card.get("checksum")})')
    print('arms       :')
    for arm in payload.get('arms', []):
        rmse = arm.get('metrics', {}).get('rmse')
        shown = f'{rmse:.6f}' if isinstance(rmse, (int, float)) else '--'
        print(f'  {arm["name"]:<20} {arm["status"]:<18} rmse={shown}')
    print('gates      :')
    gates = payload.get('gates', [])
    for gate in gates:
        mark = 'PASS' if gate['passed'] else 'FAIL'
        print(
            f'  [{mark}] {gate["name"]}: {gate["observed"]} '
            f'vs {gate["threshold"]}  ({gate["anchor"]})'
        )
    for gap in payload.get('gaps', []):
        print(f'gap        : {gap}')

    recorded = payload.get('status')
    derived = (
        STATUS_VERIFIED
        if gates and all(gate['passed'] for gate in gates)
        else payload.get('status')
    )
    print(f'status     : {recorded}')
    if recorded != derived:
        print(
            f'MISMATCH   : recorded {recorded!r} but gates imply '
            f'{derived!r} -- log a bug in VERIFICATION.md'
        )
        return 1
    return 0 if recorded == STATUS_VERIFIED else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument vector, for testing.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'run_dir',
        nargs='?',
        type=pathlib.Path,
        help='run directory to re-verify',
    )
    parser.add_argument(
        '--env',
        action='store_true',
        help='print the interpreter and package inventory',
    )
    parser.add_argument(
        '--missing',
        action='store_true',
        help='print the pip names of absent packages, space separated',
    )
    args = parser.parse_args(argv)

    if args.missing:
        return print_missing()

    if args.env or args.run_dir is None:
        code = print_environment()
        if args.run_dir is None:
            return code
    return verify_run(args.run_dir)


if __name__ == '__main__':
    sys.exit(main())
