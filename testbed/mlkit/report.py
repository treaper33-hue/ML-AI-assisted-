"""Run reports: the audit pack every experiment leaves behind.

A run directory must be enough, on its own, for a reviewer to understand and
replay the run [SKL ch.14 "What to Include in an Audit Pack"].
"""

import dataclasses
import datetime
import json
import pathlib
import platform
import sys
from typing import Any, Dict, List, Optional

from mlkit import STATUS_BUG
from mlkit import STATUS_PENDING
from mlkit import STATUS_UNVERIFIED
from mlkit import STATUS_VERIFIED

RUNS_DIRNAME = 'runs'


def _escape_cell(text: str) -> str:
    """Escapes pipes so a value cannot break a Markdown table.

    Thresholds routinely contain absolute-value bars, as in
    'max |w - w_true| < 0.1'.

    Args:
        text: Raw cell content.

    Returns:
        The content with pipes escaped.
    """
    return str(text).replace('|', r'\|')


def env_info() -> Dict[str, Any]:
    """Collects the interpreter and package versions for the report.

    Returns:
        A dictionary describing the environment. Packages that are not
        installed are recorded as 'MISSING' rather than omitted, because a
        missing package is why an arm reports UNVERIFIED (ledger GAP-13).
        A package that is installed but fails to import is recorded as
        'BROKEN (<ErrorType>)': that is a different condition, it produces
        a BUG tag rather than a skip, and it must never crash the report
        that is supposed to describe it.
    """
    packages: Dict[str, str] = {}
    for name in ('numpy', 'scipy', 'pandas', 'sklearn', 'torch', 'matplotlib'):
        try:
            module = __import__(name)
            packages[name] = getattr(module, '__version__', 'unknown')
        except ImportError:
            packages[name] = 'MISSING'
        except Exception as error:  # pylint: disable=broad-except
            packages[name] = f'BROKEN ({type(error).__name__})'
    return {
        'python': sys.version.split()[0],
        'executable': sys.executable,
        'platform': platform.platform(),
        'packages': packages,
    }


@dataclasses.dataclass
class ArmResult:
    """One methodology's result inside an experiment.

    Attributes:
        name: Arm name, matching the config.
        params: Parameters actually used.
        metrics: Metric name to value, all from `mlkit.metrics`.
        ci: Metric name to [lower, upper] confidence interval.
        status: One of the module-level status constants.
        notes: Why the status is what it is.
        seconds: Wall-clock time for the arm.
    """

    name: str
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    ci: Dict[str, List[float]] = dataclasses.field(default_factory=dict)
    status: str = STATUS_PENDING
    notes: str = ''
    seconds: float = 0.0


@dataclasses.dataclass
class Gate:
    """An acceptance check declared before the code was written.

    Attributes:
        name: Short identifier.
        anchor: Page anchor the threshold came from.
        threshold: The declared acceptance value.
        observed: What the run actually produced.
        passed: Whether the gate passed.
    """

    name: str
    anchor: str
    threshold: str
    observed: str
    passed: bool


@dataclasses.dataclass
class RunReport:
    """The full record of one experiment run.

    Attributes:
        experiment: Experiment name.
        seed: Seed used.
        config: The resolved configuration.
        dataset_card: Provenance of the data.
        splits: One description line per fold.
        arms: Per-methodology results.
        gates: Acceptance checks.
        gaps: Gap IDs that apply to this run.
        status: Overall status, derived from the gates and arms.
        created_at: ISO 8601 time the run was written, filled in by
            `write_run` from the same moment it stamps the directory
            with. It is kept inside the pack so that a reviewer holding
            `report.json` alone still knows when the run happened; the
            directory name is a convention, not data.
    """

    experiment: str
    seed: int
    config: Dict[str, Any]
    dataset_card: Dict[str, Any]
    splits: List[str] = dataclasses.field(default_factory=list)
    arms: List[ArmResult] = dataclasses.field(default_factory=list)
    gates: List[Gate] = dataclasses.field(default_factory=list)
    gaps: List[str] = dataclasses.field(default_factory=list)
    status: str = STATUS_PENDING
    created_at: str = ''

    def resolve_status(self) -> str:
        """Derives the overall status from the gates and arms.

        Returns:
            VERIFIED when every gate passed and at least one arm is
            verified; PENDING REVISION when any gate failed or any arm is
            tagged BUG, since a bug reopens the gate it touches;
            UNVERIFIED when there are no gates to judge by.
        """
        if any(arm.status == STATUS_BUG for arm in self.arms):
            self.status = STATUS_PENDING
        elif any(not gate.passed for gate in self.gates):
            self.status = STATUS_PENDING
        elif not self.gates:
            self.status = STATUS_UNVERIFIED
        elif any(arm.status == STATUS_VERIFIED for arm in self.arms):
            self.status = STATUS_VERIFIED
        else:
            self.status = STATUS_UNVERIFIED
        return self.status

    def as_dict(self) -> Dict[str, Any]:
        """Returns the report as a JSON-serialisable dictionary."""
        return {
            'experiment': self.experiment,
            'created_at': self.created_at,
            'status': self.status,
            'seed': self.seed,
            'config': self.config,
            'dataset_card': self.dataset_card,
            'splits': self.splits,
            'arms': [dataclasses.asdict(arm) for arm in self.arms],
            'gates': [dataclasses.asdict(gate) for gate in self.gates],
            'gaps': self.gaps,
            'env': env_info(),
        }

    def to_markdown(self) -> str:
        """Renders the human-readable half of the audit pack."""
        lines = [
            f'# {self.experiment}',
            '',
            f'**Status:** {self.status}  ',
            f'**Seed:** {self.seed}  ',
            f'**Dataset:** {self.dataset_card.get("name")} '
            f'(checksum `{self.dataset_card.get("checksum")}`)',
            '',
            '## Arms',
            '',
            '| arm | status | metrics | 95% CI | seconds |',
            '|---|---|---|---|---|',
        ]
        for arm in self.arms:
            metrics = ', '.join(
                f'{k}={v:.6g}' for k, v in sorted(arm.metrics.items())
            )
            intervals = ', '.join(
                f'{k}=[{v[0]:.4g}, {v[1]:.4g}]'
                for k, v in sorted(arm.ci.items())
            )
            lines.append(
                f'| {arm.name} | {arm.status} | {metrics or "--"} | '
                f'{intervals or "--"} | {arm.seconds:.3f} |'
            )
        lines += [
            '',
            '## Gates',
            '',
            '| gate | anchor | threshold | observed | passed |',
            '|---|---|---|---|---|',
        ]
        for gate in self.gates:
            mark = 'yes' if gate.passed else 'NO'
            lines.append(
                f'| {gate.name} | {_escape_cell(gate.anchor)} | '
                f'{_escape_cell(gate.threshold)} | '
                f'{_escape_cell(gate.observed)} | {mark} |'
            )
        lines += ['', '## Splits', '']
        lines += [f'- {line}' for line in self.splits] or ['- (none)']
        lines += ['', '## Known gaps affecting this run', '']
        lines += [f'- {gap}' for gap in self.gaps] or ['- (none)']
        notes = [arm for arm in self.arms if arm.notes]
        if notes:
            lines += ['', '## Arm notes', '']
            lines += [f'- **{arm.name}**: {arm.notes}' for arm in notes]
        return '\n'.join(lines) + '\n'


def write_run(
    report: RunReport,
    root: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    """Writes the audit pack for a run.

    Args:
        report: The report to persist.
        root: Testbed root; defaults to this file's grandparent.

    Returns:
        The run directory that was created.
    """
    base = root or pathlib.Path(__file__).resolve().parent.parent
    moment = datetime.datetime.now()
    stamp = moment.strftime('%Y%m%dT%H%M%S')
    if not report.created_at:
        report.created_at = moment.isoformat(timespec='seconds')
    run_dir = base / RUNS_DIRNAME / f'{stamp}-{report.experiment}'
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    (run_dir / 'report.json').write_text(
        json.dumps(payload, indent=2, default=str), encoding='utf-8'
    )
    (run_dir / 'report.md').write_text(report.to_markdown(), encoding='utf-8')
    (run_dir / 'config.json').write_text(
        json.dumps(report.config, indent=2, default=str), encoding='utf-8'
    )
    (run_dir / 'env.txt').write_text(
        json.dumps(payload['env'], indent=2), encoding='utf-8'
    )
    (run_dir / 'dataset_card.json').write_text(
        json.dumps(report.dataset_card, indent=2), encoding='utf-8'
    )
    return run_dir
