"""Parses the audit packs under `runs/` into one normalised payload.

Everything the browser sees comes from `build_payload`. The browser is
never given a raw `report.json`, for two reasons:

  * A diverged run writes `NaN` into its metrics, which is valid to
    Python's `json` module and a syntax error to the browser's
    `JSON.parse`. Divergence is a required result of this program
    [MML p.381], so the viewer has to survive it.
  * Normalising here is what lets the renderer stay dumb enough to be
    trusted: it cannot show a metric without a status tag, because a
    metric cell and its tag arrive together or not at all.

A report that cannot be read is reported as unreadable. It is never
given one of the four status tags, because the panel does not get to
decide a run's status -- only the run does.
"""

import datetime
import json
import math
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mlkit import STATUS_BUG
from mlkit import STATUS_PENDING
from mlkit import STATUS_UNVERIFIED
from mlkit import STATUS_VERIFIED
from mlkit import report as report_lib

#: Taken from the writer rather than restated, so the panel and
#: `mlkit.report.write_run` can never disagree about where runs live.
RUNS_DIRNAME = report_lib.RUNS_DIRNAME

#: The four tags a run or arm may carry. Anything else is a defect in
#: the report, not a fifth tag the panel is free to accept.
KNOWN_STATUSES: Tuple[str, ...] = (
    STATUS_VERIFIED,
    STATUS_PENDING,
    STATUS_UNVERIFIED,
    STATUS_BUG,
)

#: Shown when a report cannot be parsed. Deliberately not a member of
#: KNOWN_STATUSES: it is a parse outcome, not a verification verdict.
UNREADABLE = 'UNREADABLE'

#: Above this magnitude a metric is reported as diverged rather than
#: plotted, so one 1e+65 cannot flatten every other point on an axis.
#: Same rule as `run_sweep` in experiments/exp01_baseline.py.
DIVERGENCE_LIMIT = 1e6

#: Points whose spread is below this, relative to their magnitude, are
#: the same point. Float noise lands around 1e-16, while anything a sweep
#: is meant to reveal is orders of magnitude larger. Drawing a slope
#: through the gap between them would turn rounding error into a trend --
#: the pretty-but-wrong picture this testbed exists to catch.
COLLAPSE_RELATIVE_SPREAD = 1e-9

#: Intervals on the y-axis, so one more label than this is drawn.
AXIS_TICK_INTERVALS = 4

#: Significant figures tried for tick labels, lowest first. Labels must
#: distinguish the ticks they sit beside: two ticks printing the same
#: text means the axis is claiming a resolution it does not have.
MIN_TICK_DIGITS = 4
MAX_TICK_DIGITS = 17

#: Significant figures for the single label on a collapsed axis.
COLLAPSED_TICK_DIGITS = 6

#: Config keys that record what was changed rather than what was run.
#: They are excluded when deciding whether two configs differ.
BOOKKEEPING_KEYS = frozenset({'overrides'})

RUN_STAMP_FORMAT = '%Y%m%dT%H%M%S'
REPORT_FILENAME = 'report.json'

Json = Dict[str, Any]


def _is_number(value: Any) -> bool:
    """Reports whether a value is a real number rather than a flag.

    Args:
        value: Any decoded JSON value.

    Returns:
        True for int and float, False for bool -- `True` is an int in
        Python and would otherwise be charted as the number 1.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any) -> Optional[float]:
    """Coerces a reported value to a float when that is meaningful.

    `report.json` is written with `json.dumps(..., default=str)`, so a
    value the encoder could not handle arrives as its string form. A
    numeric string is therefore a number that lost its type on the way
    to disk, not a label.

    Args:
        value: Any decoded JSON value.

    Returns:
        The value as a float, or None when it is not numeric.
    """
    if _is_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def metric_cell(value: Any) -> Json:
    """Normalises one metric into a JSON-safe cell for the renderer.

    Divergence is a result of this program, not an error [MML p.381],
    so a non-finite or enormous value is labelled `diverged` instead of
    being dropped or plotted. Only a usable number carries a numeric
    `value`, which is what keeps a diverged point off the axis.

    Args:
        value: Raw metric value from the report; may be missing,
            non-finite or non-numeric.

    Returns:
        A dictionary with keys:
            value: The float to plot, or None when there is none.
            text: What to print, never an empty string.
            diverged: Whether the value left the usable range.
            present: Whether the report carried a value at all.
    """
    if value is None:
        return {
            'value': None,
            'text': '--',
            'diverged': False,
            'present': False,
        }
    number = _as_float(value)
    if number is None:
        return {
            'value': None,
            'text': f'(not numeric: {value!r})',
            'diverged': False,
            'present': True,
        }
    if math.isnan(number):
        text = 'diverged (nan)'
    elif math.isinf(number):
        text = 'diverged (inf)' if number > 0 else 'diverged (-inf)'
    elif abs(number) > DIVERGENCE_LIMIT:
        text = f'diverged ({number:.2e})'
    else:
        return {
            'value': number,
            'text': f'{number:.6g}',
            'diverged': False,
            'present': True,
        }
    return {
        'value': None,
        'text': text,
        'diverged': True,
        'present': True,
    }


def ci_cell(interval: Any) -> Json:
    """Normalises one confidence interval into a display cell.

    Args:
        interval: Raw interval from the report, expected to be a
            two-element sequence of numbers.

    Returns:
        A dictionary with keys 'lower', 'upper', 'text', 'present' and
        'diverged'. A malformed interval is reported as malformed
        rather than being silently halved or dropped.
    """
    if interval is None:
        return {
            'lower': None,
            'upper': None,
            'text': '--',
            'present': False,
            'diverged': False,
        }
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return {
            'lower': None,
            'upper': None,
            'text': '(malformed interval)',
            'present': True,
            'diverged': False,
        }
    low, high = (metric_cell(bound) for bound in interval)
    if low['value'] is None or high['value'] is None:
        return {
            'lower': None,
            'upper': None,
            'text': 'diverged',
            'present': True,
            'diverged': True,
        }
    return {
        'lower': low['value'],
        'upper': high['value'],
        'text': f'[{low["value"]:.4g}, {high["value"]:.4g}]',
        'present': True,
        'diverged': False,
    }


def flatten_config(config: Any, prefix: str = '') -> Dict[str, Any]:
    """Flattens a nested config into the dotted keys the CLI uses.

    The dotted form is what `--set` and `--sweep` already speak, so a
    difference the panel shows can be pasted straight back into a
    command. An empty table is kept as a leaf, because
    `[arms.normal_equations]` declares an arm and losing it would hide
    a real difference between two configs.

    Args:
        config: A configuration mapping, or any nested value.
        prefix: Dotted prefix accumulated by the recursion.

    Returns:
        A mapping from dotted key to leaf value. Returns an empty
        mapping when `config` is not a mapping.
    """
    if not isinstance(config, dict):
        return {}
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        dotted = f'{prefix}{key}'
        if isinstance(value, dict) and value:
            flat.update(flatten_config(value, f'{dotted}.'))
        else:
            flat[dotted] = value
    return flat


def _comparable(value: Any) -> str:
    """Renders a config value as a stable string for equality tests.

    Args:
        value: Any decoded JSON value.

    Returns:
        A canonical string form, so lists and tables compare by content
        rather than by identity.
    """
    return json.dumps(value, sort_keys=True, default=str)


def _is_bookkeeping(dotted: str) -> bool:
    """Reports whether a dotted key records provenance, not a setting.

    Args:
        dotted: A dotted config key.

    Returns:
        True when the key is bookkeeping such as 'overrides'.
    """
    return dotted.split('.')[0] in BOOKKEEPING_KEYS


def timestamp_from_run_id(run_id: str) -> Optional[str]:
    """Recovers the run time from the directory name.

    `mlkit.report.write_run` names a run `<stamp>-<experiment>`, so the
    directory carries the time even for reports written before
    `created_at` existed.

    Args:
        run_id: Directory name of a run.

    Returns:
        An ISO 8601 timestamp, or None when the name does not start
        with a stamp this function understands.
    """
    stamp = run_id.split('-', 1)[0]
    try:
        moment = datetime.datetime.strptime(stamp, RUN_STAMP_FORMAT)
    except ValueError:
        return None
    return moment.isoformat()


def _resolve_timestamp(payload: Json, run_id: str) -> Tuple[str, str]:
    """Decides which timestamp to show and says where it came from.

    Args:
        payload: The decoded report.
        run_id: Directory name of the run.

    Returns:
        A tuple of (ISO timestamp or '', source) where source is
        'report', 'directory' or 'unknown'.
    """
    reported = payload.get('created_at')
    if isinstance(reported, str) and reported.strip():
        return reported.strip(), 'report'
    derived = timestamp_from_run_id(run_id)
    if derived is not None:
        return derived, 'directory'
    return '', 'unknown'


def _normalise_status(raw: Any, errors: List[str], label: str) -> Json:
    """Normalises a status field without ever supplying a missing tag.

    Args:
        raw: The status value as found in the report.
        errors: Error list for this run, appended to in place.
        label: What the status belongs to, for the error message.

    Returns:
        A dictionary with keys 'status' (the tag or None) and 'known'
        (whether it is one of the four).
    """
    if not isinstance(raw, str) or not raw.strip():
        errors.append(f'{label}: no status tag in the report')
        return {'status': None, 'known': False}
    tag = raw.strip()
    if tag not in KNOWN_STATUSES:
        errors.append(f'{label}: unrecognised status tag {tag!r}')
        return {'status': tag, 'known': False}
    return {'status': tag, 'known': True}


def _normalise_arm(raw: Any, index: int, errors: List[str]) -> Json:
    """Normalises one arm entry from the report.

    Args:
        raw: The arm entry, expected to be a mapping.
        index: Position in the arms list, used when the name is absent.
        errors: Error list for this run, appended to in place.

    Returns:
        The normalised arm. Metrics are cells, so the renderer always
        has the arm's tag in hand when it prints a number.
    """
    if not isinstance(raw, dict):
        errors.append(f'arm #{index}: entry is not an object')
        return {
            'name': f'(arm #{index})',
            'params': {},
            'metrics': {},
            'ci': {},
            'status': None,
            'known_status': False,
            'notes': '',
            'seconds': None,
            'present': False,
            'declared': False,
        }
    name = raw.get('name')
    if not isinstance(name, str) or not name.strip():
        errors.append(f'arm #{index}: no name in the report')
        name = f'(arm #{index})'
    tag = _normalise_status(raw.get('status'), errors, f'arm {name}')
    metrics_raw = raw.get('metrics')
    if metrics_raw is None:
        metrics_raw = {}
    elif not isinstance(metrics_raw, dict):
        errors.append(f'arm {name}: metrics is not an object')
        metrics_raw = {}
    ci_raw = raw.get('ci')
    if not isinstance(ci_raw, dict):
        if ci_raw is not None:
            errors.append(f'arm {name}: ci is not an object')
        ci_raw = {}
    seconds = _as_float(raw.get('seconds'))
    notes = raw.get('notes')
    params = raw.get('params')
    return {
        'name': name,
        'params': params if isinstance(params, dict) else {},
        'metrics': {
            key: metric_cell(value)
            for key, value in sorted(metrics_raw.items())
        },
        'ci': {key: ci_cell(value) for key, value in sorted(ci_raw.items())},
        'status': tag['status'],
        'known_status': tag['known'],
        'notes': notes if isinstance(notes, str) else '',
        'seconds': seconds,
        'present': True,
        'declared': True,
    }


def _absent_arm(name: str) -> Json:
    """Builds the row for an arm the config declared but did not report.

    Args:
        name: Arm name taken from the config.

    Returns:
        An arm record marked absent, carrying no metrics and no tag.
    """
    return {
        'name': name,
        'params': {},
        'metrics': {},
        'ci': {},
        'status': None,
        'known_status': False,
        'notes': 'declared in the config, absent from the report',
        'seconds': None,
        'present': False,
        'declared': True,
    }


def _normalise_gate(raw: Any, index: int, errors: List[str]) -> Json:
    """Normalises one acceptance gate from the report.

    Args:
        raw: The gate entry, expected to be a mapping.
        index: Position in the gates list, for error messages.
        errors: Error list for this run, appended to in place.

    Returns:
        The normalised gate. `passed` stays None when the report did
        not say, because an unanswered gate is not a passed one.
    """
    if not isinstance(raw, dict):
        errors.append(f'gate #{index}: entry is not an object')
        return {
            'name': f'(gate #{index})',
            'anchor': '',
            'threshold': '',
            'observed': '',
            'passed': None,
        }
    passed = raw.get('passed')
    if not isinstance(passed, bool):
        errors.append(f'gate #{index}: no pass/fail in the report')
        passed = None
    return {
        'name': str(raw.get('name', f'(gate #{index})')),
        'anchor': str(raw.get('anchor', '')),
        'threshold': str(raw.get('threshold', '')),
        'observed': str(raw.get('observed', '')),
        'passed': passed,
    }


def _string_list(raw: Any) -> List[str]:
    """Coerces a reported list into a list of strings.

    Args:
        raw: The reported value.

    Returns:
        A list of strings; empty when the value was not a sequence.
    """
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item) for item in raw]


def parse_report(payload: Any, run_id: str) -> Json:
    """Normalises one decoded `report.json` into a panel run record.

    Args:
        payload: The decoded report, expected to be a mapping.
        run_id: Directory name of the run, used as its identity.

    Returns:
        A run record. `readable` is False when the payload was not a
        report at all; otherwise every field the report omitted is
        None or empty and named in `errors`.
    """
    if not isinstance(payload, dict):
        return unreadable_run(run_id, 'report.json is not a JSON object')

    errors: List[str] = []
    experiment = payload.get('experiment')
    if not isinstance(experiment, str) or not experiment.strip():
        errors.append('no experiment name in the report')
        experiment = '(unnamed)'
    overall = _normalise_status(payload.get('status'), errors, 'run')
    config = payload.get('config')
    if not isinstance(config, dict):
        errors.append('no config in the report')
        config = {}
    card = payload.get('dataset_card')
    if not isinstance(card, dict):
        errors.append('no dataset card in the report')
        card = {}
    env = payload.get('env')
    if not isinstance(env, dict):
        errors.append('no environment block in the report')
        env = {}

    arms_raw = payload.get('arms')
    if not isinstance(arms_raw, list):
        if arms_raw is not None:
            errors.append('arms is not a list')
        else:
            errors.append('no arms in the report')
        arms_raw = []
    arms = [
        _normalise_arm(entry, index, errors)
        for index, entry in enumerate(arms_raw)
    ]
    reported = {arm['name'] for arm in arms}
    declared = [
        name for name in flatten_declared_arms(config) if name not in reported
    ]
    for name in declared:
        errors.append(f'arm {name}: declared in the config but not reported')
        arms.append(_absent_arm(name))

    gates_raw = payload.get('gates')
    if not isinstance(gates_raw, list):
        if gates_raw is not None:
            errors.append('gates is not a list')
        gates_raw = []
    gates = [
        _normalise_gate(entry, index, errors)
        for index, entry in enumerate(gates_raw)
    ]

    primary = config.get('metric', {})
    primary_metric = (
        primary.get('primary') if isinstance(primary, dict) else None
    )
    if not isinstance(primary_metric, str) or not primary_metric.strip():
        errors.append('no primary metric declared in the config')
        primary_metric = None

    seed = payload.get('seed')
    if not _is_number(seed):
        errors.append('no seed in the report')
        seed = None
    timestamp, source = _resolve_timestamp(payload, run_id)
    checksum = card.get('checksum')

    return {
        'run_id': run_id,
        'readable': True,
        'experiment': experiment.strip(),
        'timestamp': timestamp,
        'timestamp_source': source,
        'seed': seed,
        'status': overall['status'],
        'known_status': overall['known'],
        'dataset_checksum': (
            str(checksum) if isinstance(checksum, (str, int)) else None
        ),
        'dataset_card': card,
        'primary_metric': primary_metric,
        'config': config,
        'config_flat': {
            key: value
            for key, value in flatten_config(config).items()
            if not _is_bookkeeping(key)
        },
        'overrides': _string_list(config.get('overrides')),
        'splits': _string_list(payload.get('splits')),
        'arms': arms,
        'missing_arms': declared,
        'gates': gates,
        'gaps': _string_list(payload.get('gaps')),
        'env': env,
        'errors': errors,
    }


def flatten_declared_arms(config: Json) -> List[str]:
    """Lists the arm names a config declares.

    Args:
        config: A resolved experiment configuration.

    Returns:
        The declared arm names in config order; empty when the config
        declares no arms table.
    """
    arms = config.get('arms')
    if not isinstance(arms, dict):
        return []
    return [str(name) for name in arms]


def unreadable_run(run_id: str, reason: str) -> Json:
    """Builds the record for a run whose report could not be read.

    The record carries no status tag. UNREADABLE is a parse outcome and
    the panel does not get to decide a run's verification status.

    Args:
        run_id: Directory name of the run.
        reason: What went wrong, shown to the reader verbatim.

    Returns:
        A run record with `readable` False and empty tables.
    """
    timestamp = timestamp_from_run_id(run_id)
    return {
        'run_id': run_id,
        'readable': False,
        'experiment': run_id.split('-', 1)[-1],
        'timestamp': timestamp or '',
        'timestamp_source': 'directory' if timestamp else 'unknown',
        'seed': None,
        'status': None,
        'known_status': False,
        'dataset_checksum': None,
        'dataset_card': {},
        'primary_metric': None,
        'config': {},
        'config_flat': {},
        'overrides': [],
        'splits': [],
        'arms': [],
        'missing_arms': [],
        'gates': [],
        'gaps': [],
        'env': {},
        'errors': [reason],
    }


def load_run(run_dir: pathlib.Path) -> Optional[Json]:
    """Loads and normalises one run directory.

    Args:
        run_dir: A directory under `runs/`.

    Returns:
        The run record, or None when the directory holds no
        `report.json` at all -- a run that was interrupted before it
        wrote one is not a run to review.
    """
    report_path = run_dir / REPORT_FILENAME
    if not report_path.is_file():
        return None
    try:
        text = report_path.read_text(encoding='utf-8')
    except OSError as error:
        return unreadable_run(run_dir.name, f'cannot read: {error}')
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        return unreadable_run(
            run_dir.name, f'malformed JSON at line {error.lineno}: {error.msg}'
        )
    return parse_report(payload, run_dir.name)


def load_runs(runs_root: pathlib.Path) -> List[Json]:
    """Loads every readable run directory, newest first.

    Args:
        runs_root: The `runs/` directory.

    Returns:
        Run records sorted by directory name descending, which is
        newest first given the timestamp prefix. An absent or empty
        directory yields an empty list rather than an error: a testbed
        with no runs yet is a normal state.
    """
    if not runs_root.is_dir():
        return []
    runs: List[Json] = []
    for entry in sorted(runs_root.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        record = load_run(entry)
        if record is not None:
            runs.append(record)
    return runs


def unverified_runs(runs: Sequence[Json]) -> List[Json]:
    """Selects the runs that are not VERIFIED.

    These are the runs the banner names. They are never filtered out of
    the board: an unverified result that is out of sight is the failure
    mode this program exists to prevent.

    Args:
        runs: Run records.

    Returns:
        The subset whose status is anything other than VERIFIED,
        including runs that could not be read.
    """
    return [run for run in runs if run.get('status') != STATUS_VERIFIED]


def primary_cell(run: Json, arm_name: Optional[str] = None) -> Json:
    """Finds a run's primary metric for one arm.

    Args:
        run: A run record.
        arm_name: Arm to read; when None, the first reported arm that
            carries the primary metric is used.

    Returns:
        A dictionary with keys 'arm', 'arm_status', 'known_status',
        'metric' (the metric name or None) and 'cell' (a metric cell).
        The arm's tag is part of the return value so that no caller can
        obtain the number without it.
    """
    metric = run.get('primary_metric')
    candidates: Iterable[Json] = run.get('arms', [])
    if arm_name is not None:
        candidates = [arm for arm in candidates if arm.get('name') == arm_name]
    chosen: Optional[Json] = None
    for arm in candidates:
        chosen = chosen or arm
        if metric and metric in arm.get('metrics', {}):
            chosen = arm
            break
    if chosen is None:
        return {
            'arm': arm_name,
            'arm_status': None,
            'known_status': False,
            'metric': metric,
            'cell': metric_cell(None),
        }
    cell = chosen.get('metrics', {}).get(metric) if metric else None
    return {
        'arm': chosen.get('name'),
        'arm_status': chosen.get('status'),
        'known_status': bool(chosen.get('known_status')),
        'metric': metric,
        'cell': cell or metric_cell(None),
    }


def _axis_sort_key(value: Any) -> Tuple[int, float, str]:
    """Orders sweep points numerically when it can, textually otherwise.

    Args:
        value: The swept config value.

    Returns:
        A sort key placing numbers before strings.
    """
    number = _as_float(value)
    if number is None or math.isnan(number):
        return (1, 0.0, _comparable(value))
    return (0, number, '')


def sweep_groups(runs: Sequence[Json]) -> List[Json]:
    """Finds sets of runs that differ in exactly one config value.

    This is what makes a parameter change traceable to a metric
    change: every other key is held equal by construction, so the one
    key that moved is the only candidate explanation. Runs that differ
    in two keys form no group, because such a pair explains nothing.

    Args:
        runs: Run records, readable or not.

    Returns:
        One group per (experiment, swept key) with at least two
        distinct values, each group's points sorted by axis value.
    """
    usable = [
        run for run in runs if run.get('readable') and run.get('config_flat')
    ]
    by_experiment: Dict[str, List[Json]] = {}
    for run in usable:
        by_experiment.setdefault(str(run.get('experiment')), []).append(run)

    groups: List[Json] = []
    for experiment, members in sorted(by_experiment.items()):
        keys = sorted({key for run in members for key in run['config_flat']})
        for key in keys:
            buckets: Dict[Tuple[Any, ...], List[Json]] = {}
            for run in members:
                flat = run['config_flat']
                if key not in flat:
                    continue
                signature = tuple(
                    (other, _comparable(flat[other]))
                    for other in sorted(flat)
                    if other != key
                )
                buckets.setdefault(signature, []).append(run)
            for members_here in buckets.values():
                group = _build_sweep_group(experiment, key, members_here)
                if group is not None:
                    groups.append(group)
    groups.sort(key=lambda group: (group['experiment'], group['key']))
    return groups


def format_significant(value: float, digits: int) -> str:
    """Formats a number to a fixed number of significant figures.

    Trailing zeros are kept, so a column of tick labels lines up instead
    of ragging between '0.562' and '0.05333'.

    Args:
        value: The number to format.
        digits: Significant figures to show; at least 1.

    Returns:
        The formatted number, in scientific notation only when plain
        notation would be unreadable.

    Raises:
        ValueError: If `digits` is less than 1.
    """
    if digits < 1:
        raise ValueError('digits must be >= 1')
    if value == 0.0 or not math.isfinite(value):
        return f'{value:.{digits - 1}f}'
    exponent = math.floor(math.log10(abs(value)))
    if exponent < -6 or exponent >= digits:
        return f'{value:.{digits - 1}e}'
    return f'{value:.{max(digits - 1 - exponent, 0)}f}'


def tick_digits(values: Sequence[float]) -> int:
    """Finds the fewest significant figures that tell the ticks apart.

    Args:
        values: Tick values, in any order.

    Returns:
        The smallest digit count in [MIN_TICK_DIGITS, MAX_TICK_DIGITS]
        that gives every value a distinct label, or MAX_TICK_DIGITS when
        no count does.
    """
    for digits in range(MIN_TICK_DIGITS, MAX_TICK_DIGITS + 1):
        labels = [format_significant(value, digits) for value in values]
        if len(set(labels)) == len(labels):
            return digits
    return MAX_TICK_DIGITS


def axis_scale(values: Sequence[Optional[float]]) -> Json:
    """Decides the y-scale and tick labels for a sweep chart.

    This lives here rather than in the renderer because it is a claim
    about the numbers, not about pixels: whether two points differ at
    all, and how many figures the axis may honestly show. The browser
    maps the result to coordinates and nothing more.

    Args:
        values: Metric values in point order; None for a point that
            diverged or is missing.

    Returns:
        A dictionary with `collapsed`, `common_value`, `relative_spread`,
        the `lo`/`hi` bounds of the drawing area, and `ticks`, a list of
        `{value, label}` entries.
    """
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(value)
    ]
    if not finite:
        return {
            'collapsed': False,
            'common_value': None,
            'relative_spread': None,
            'lo': 0.0,
            'hi': 1.0,
            'ticks': [],
        }

    low, high = min(finite), max(finite)
    magnitude = max(abs(low), abs(high))
    spread = high - low
    relative = spread / magnitude if magnitude > 0 else 0.0
    if spread == 0.0 or relative < COLLAPSE_RELATIVE_SPREAD:
        # Centre the shared value and give it room on both sides, so the
        # points sit on one flat line instead of being stretched apart.
        band = magnitude * 0.5 or 1.0
        return {
            'collapsed': True,
            'common_value': high,
            'relative_spread': relative,
            'lo': low - band,
            'hi': high + band,
            'ticks': [
                {
                    'value': high,
                    'label': format_significant(high, COLLAPSED_TICK_DIGITS),
                }
            ],
        }

    pad = spread * 0.12
    lo, hi = low - pad, high + pad
    raw = [
        lo + (hi - lo) * index / AXIS_TICK_INTERVALS
        for index in range(AXIS_TICK_INTERVALS + 1)
    ]
    digits = tick_digits(raw)
    return {
        'collapsed': False,
        'common_value': None,
        'relative_spread': relative,
        'lo': lo,
        'hi': hi,
        'ticks': [
            {'value': value, 'label': format_significant(value, digits)}
            for value in raw
        ],
    }


def _build_sweep_group(
    experiment: str,
    key: str,
    members: Sequence[Json],
) -> Optional[Json]:
    """Builds one sweep group, or None when there is nothing to plot.

    Args:
        experiment: Experiment name shared by the members.
        key: The dotted config key that varies.
        members: Runs that agree on every other config key.

    Returns:
        The group, or None when fewer than two distinct values of `key`
        are present.
    """
    values = {_comparable(run['config_flat'][key]) for run in members}
    if len(members) < 2 or len(values) < 2:
        return None
    parts = key.split('.')
    arm = parts[1] if parts[0] == 'arms' and len(parts) > 2 else None
    points: List[Json] = []
    for run in members:
        axis_value = run['config_flat'][key]
        found = primary_cell(run, arm)
        points.append(
            {
                'run_id': run['run_id'],
                'axis_value': axis_value if _is_number(axis_value) else None,
                'axis_text': str(axis_value),
                'arm': found['arm'],
                'arm_status': found['arm_status'],
                'known_status': found['known_status'],
                'run_status': run.get('status'),
                'metric': found['cell'],
            }
        )
    points.sort(key=lambda point: _axis_sort_key(point['axis_text']))
    metric_names = {
        run['primary_metric'] for run in members if run.get('primary_metric')
    }
    return {
        'experiment': experiment,
        'key': key,
        'arm': arm,
        'metric': sorted(metric_names)[0] if metric_names else None,
        'points': points,
        'diverged': sum(1 for point in points if point['metric']['diverged']),
        'axis': axis_scale([point['metric'].get('value') for point in points]),
    }


def build_payload(runs_root: pathlib.Path) -> Json:
    """Builds the whole payload the browser renders.

    Args:
        runs_root: The `runs/` directory to read.

    Returns:
        A JSON-safe dictionary with the run records, the unverified
        list behind the banner, the detected sweeps, and any notice the
        reader needs in order to trust what is on screen.
    """
    runs = load_runs(runs_root)
    notices: List[str] = []
    if not runs_root.is_dir():
        notices.append(
            f'no runs directory at {runs_root}; run an experiment first'
        )
    elif not runs:
        notices.append(
            f'no readable run under {runs_root}; run an experiment first'
        )
    broken = [run['run_id'] for run in runs if not run['readable']]
    if broken:
        notices.append(
            f'{len(broken)} run(s) could not be parsed: {", ".join(broken)}'
        )
    return {
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'runs_root': str(runs_root),
        'statuses': list(KNOWN_STATUSES),
        'unreadable_label': UNREADABLE,
        'divergence_limit': DIVERGENCE_LIMIT,
        'collapse_relative_spread': COLLAPSE_RELATIVE_SPREAD,
        'runs': runs,
        'unverified': [run['run_id'] for run in unverified_runs(runs)],
        'sweeps': sweep_groups(runs),
        'notices': notices,
    }
