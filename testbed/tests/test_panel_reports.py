"""Contract tests for the control panel's report-parsing layer.

The panel is the one place in this program where a number is shown to
somebody who did not run the experiment. These tests encode what it is
allowed to do with a report it cannot fully trust:

  * never invent a status tag,
  * never drop a run quietly,
  * never hand the browser a number it could plot on an axis when the
    run that produced it diverged.

If one of these fails, the dashboard is capable of showing a result
more confidently than the run earned, which is the failure mode the
whole program is built around.
"""

import json
import pathlib

import pytest

from mlkit import STATUS_PENDING
from mlkit import STATUS_UNVERIFIED
from mlkit import STATUS_VERIFIED
from panel import reports


def make_report(**overrides):
    """Builds a complete, well-formed report payload to vary from."""
    payload = {
        'experiment': 'exp01_baseline',
        'created_at': '2026-01-18T09:30:00',
        'status': STATUS_VERIFIED,
        'seed': 20260101,
        'config': {
            'experiment': 'exp01_baseline',
            'seed': 20260101,
            'dataset': {'name': 'synthetic_linear', 'noise': 0.25},
            'metric': {'primary': 'rmse', 'secondary': ['mae']},
            'arms': {
                'normal_equations': {},
                'gradient_descent': {'lr': 0.05, 'iters': 2000},
            },
        },
        'dataset_card': {
            'name': 'synthetic_linear',
            'checksum': 'b78728653426a819',
        },
        'splits': ['random: train n=1600 test n=400'],
        'arms': [
            {
                'name': 'normal_equations',
                'params': {},
                'metrics': {'rmse': 0.2563724945, 'mae': 0.204399},
                'ci': {'rmse': [0.2401, 0.2735]},
                'status': STATUS_VERIFIED,
                'notes': '',
                'seconds': 0.00013,
            },
            {
                'name': 'gradient_descent',
                'params': {'lr': 0.05, 'iters': 2000},
                'metrics': {'rmse': 0.2563724945},
                'ci': {'rmse': [0.2401, 0.2735]},
                'status': STATUS_VERIFIED,
                'notes': '',
                'seconds': 0.0266,
            },
        ],
        'gates': [
            {
                'name': 'closed_form_recovers_truth',
                'anchor': '[MML p.210]',
                'threshold': 'max |w - w_true| < 0.1',
                'observed': '0.0120679',
                'passed': True,
            }
        ],
        'gaps': ['GAP-12: runtime unverified on this hardware'],
        'env': {'python': '3.12.11', 'packages': {'numpy': '2.3.2'}},
    }
    payload.update(overrides)
    return payload


def write_run(root: pathlib.Path, run_id: str, payload, raw=None):
    """Writes one run directory and returns its path."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    text = raw if raw is not None else json.dumps(payload, indent=2)
    (run_dir / 'report.json').write_text(text, encoding='utf-8')
    return run_dir


# ---- the happy path -------------------------------------------------


def test_parses_a_complete_report(tmp_path):
    """Every field the status board and run detail need is present."""
    write_run(tmp_path, '20260118T093000-exp01_baseline', make_report())
    (run,) = reports.load_runs(tmp_path)
    assert run['readable'] is True
    assert run['experiment'] == 'exp01_baseline'
    assert run['status'] == STATUS_VERIFIED
    assert run['known_status'] is True
    assert run['seed'] == 20260101
    assert run['dataset_checksum'] == 'b78728653426a819'
    assert run['primary_metric'] == 'rmse'
    assert run['errors'] == []
    assert run['splits'] == ['random: train n=1600 test n=400']
    assert run['gaps'][0].startswith('GAP-12')
    assert run['gates'][0]['anchor'] == '[MML p.210]'
    assert run['gates'][0]['passed'] is True
    assert run['env']['python'] == '3.12.11'


def test_every_arm_metric_arrives_with_its_tag(tmp_path):
    """A metric cell is never separable from the arm's status."""
    write_run(tmp_path, '20260118T093000-exp01_baseline', make_report())
    (run,) = reports.load_runs(tmp_path)
    for arm in run['arms']:
        assert arm['status'] in reports.KNOWN_STATUSES
        for cell in arm['metrics'].values():
            assert set(cell) == {'value', 'text', 'diverged', 'present'}
    found = reports.primary_cell(run, 'gradient_descent')
    assert found['arm_status'] == STATUS_VERIFIED
    assert found['cell']['value'] == pytest.approx(0.2563724945)


def test_created_at_is_preferred_over_the_directory_name(tmp_path):
    """The pack describes itself; the directory name is only a fallback."""
    write_run(tmp_path, '20991231T235959-exp01_baseline', make_report())
    (run,) = reports.load_runs(tmp_path)
    assert run['timestamp'] == '2026-01-18T09:30:00'
    assert run['timestamp_source'] == 'report'


def test_timestamp_falls_back_to_the_directory_name(tmp_path):
    """Reports written before created_at existed still carry a time."""
    payload = make_report()
    del payload['created_at']
    write_run(tmp_path, '20260118T093000-exp01_baseline', payload)
    (run,) = reports.load_runs(tmp_path)
    assert run['timestamp'] == '2026-01-18T09:30:00'
    assert run['timestamp_source'] == 'directory'


def test_timestamp_is_unknown_when_nothing_carries_one(tmp_path):
    """An unparseable directory name is said to be unknown, not guessed."""
    payload = make_report()
    del payload['created_at']
    write_run(tmp_path, 'handcopied-run', payload)
    (run,) = reports.load_runs(tmp_path)
    assert run['timestamp'] == ''
    assert run['timestamp_source'] == 'unknown'


# ---- reports the panel cannot trust ---------------------------------


def test_malformed_report_json_is_unreadable_and_never_tagged(tmp_path):
    """A report that will not parse gets no status tag at all.

    UNREADABLE is a parse outcome. Giving such a run one of the four
    tags would be the panel deciding a verification result, which is
    not the panel's to decide.
    """
    write_run(
        tmp_path,
        '20260118T093000-exp01_baseline',
        None,
        raw='{"experiment": "exp01_baseline", "status": ',
    )
    (run,) = reports.load_runs(tmp_path)
    assert run['readable'] is False
    assert run['status'] is None
    assert run['known_status'] is False
    assert run['arms'] == []
    assert len(run['errors']) == 1
    assert 'malformed JSON' in run['errors'][0]
    assert reports.UNREADABLE not in reports.KNOWN_STATUSES


def test_unreadable_run_is_banner_material(tmp_path):
    """An unreadable run is not VERIFIED, so the banner must name it."""
    write_run(tmp_path, '20260118T093000-exp01_baseline', None, raw='nope')
    write_run(tmp_path, '20260118T093001-exp01_baseline', make_report())
    payload = reports.build_payload(tmp_path)
    assert payload['unverified'] == ['20260118T093000-exp01_baseline']
    assert any('could not be parsed' in n for n in payload['notices'])


def test_report_that_is_not_an_object_is_unreadable(tmp_path):
    """Valid JSON that is not a report is still not a report."""
    write_run(tmp_path, '20260118T093000-exp01_baseline', None, raw='[1, 2]')
    (run,) = reports.load_runs(tmp_path)
    assert run['readable'] is False
    assert 'not a JSON object' in run['errors'][0]


def test_directory_without_a_report_is_skipped(tmp_path):
    """An interrupted run wrote no report, so there is nothing to review."""
    (tmp_path / '20260118T093000-exp01_baseline').mkdir()
    write_run(tmp_path, '20260118T093001-exp01_baseline', make_report())
    runs = reports.load_runs(tmp_path)
    assert [run['run_id'] for run in runs] == ['20260118T093001-exp01_baseline']


def test_missing_runs_directory_is_a_notice_not_a_crash(tmp_path):
    """A testbed with no runs yet is a normal state, not an error."""
    missing = tmp_path / 'runs'
    assert reports.load_runs(missing) == []
    payload = reports.build_payload(missing)
    assert payload['runs'] == []
    assert any('no runs directory' in n for n in payload['notices'])


# ---- missing pieces inside a readable report -------------------------


def test_arm_declared_in_config_but_absent_from_the_report(tmp_path):
    """A dropped arm is shown as dropped, never silently omitted."""
    payload = make_report()
    payload['arms'] = [payload['arms'][0]]
    write_run(tmp_path, '20260118T093000-exp01_baseline', payload)
    (run,) = reports.load_runs(tmp_path)
    assert run['missing_arms'] == ['gradient_descent']
    absent = [arm for arm in run['arms'] if arm['name'] == 'gradient_descent']
    assert len(absent) == 1
    assert absent[0]['present'] is False
    assert absent[0]['status'] is None
    assert absent[0]['metrics'] == {}
    assert any('gradient_descent' in err for err in run['errors'])


def test_report_with_no_arms_at_all(tmp_path):
    """No arms means no measurements, and the report says so."""
    payload = make_report()
    del payload['arms']
    write_run(tmp_path, '20260118T093000-exp01_baseline', payload)
    (run,) = reports.load_runs(tmp_path)
    assert run['readable'] is True
    assert sorted(run['missing_arms']) == [
        'gradient_descent',
        'normal_equations',
    ]
    assert all(arm['present'] is False for arm in run['arms'])
    assert 'no arms in the report' in run['errors']


def test_untagged_arm_keeps_its_number_and_is_marked_untagged(tmp_path):
    """An untagged metric is shown loudly, not hidden from the reviewer."""
    payload = make_report()
    del payload['arms'][0]['status']
    write_run(tmp_path, '20260118T093000-exp01_baseline', payload)
    (run,) = reports.load_runs(tmp_path)
    arm = run['arms'][0]
    assert arm['status'] is None
    assert arm['known_status'] is False
    assert arm['metrics']['rmse']['value'] == pytest.approx(0.2563724945)
    assert any('no status tag' in err for err in run['errors'])


def test_unrecognised_status_is_not_promoted_to_a_tag(tmp_path):
    """A fifth tag is a defect in the report, not a tag the panel takes."""
    payload = make_report(status='LOOKS FINE')
    write_run(tmp_path, '20260118T093000-exp01_baseline', payload)
    (run,) = reports.load_runs(tmp_path)
    assert run['status'] == 'LOOKS FINE'
    assert run['known_status'] is False
    assert any('unrecognised status' in err for err in run['errors'])


def test_gate_without_a_verdict_is_not_a_passed_gate(tmp_path):
    """A gate the report did not answer stays unanswered."""
    payload = make_report()
    del payload['gates'][0]['passed']
    write_run(tmp_path, '20260118T093000-exp01_baseline', payload)
    (run,) = reports.load_runs(tmp_path)
    assert run['gates'][0]['passed'] is None
    assert any('pass/fail' in err for err in run['errors'])


def test_missing_config_and_card_are_reported_not_filled_in(tmp_path):
    """Absent blocks come back empty and named, never defaulted."""
    payload = make_report()
    del payload['config']
    del payload['dataset_card']
    del payload['seed']
    write_run(tmp_path, '20260118T093000-exp01_baseline', payload)
    (run,) = reports.load_runs(tmp_path)
    assert run['config'] == {}
    assert run['dataset_checksum'] is None
    assert run['seed'] is None
    assert run['primary_metric'] is None
    assert 'no config in the report' in run['errors']
    assert 'no seed in the report' in run['errors']


# ---- divergence ------------------------------------------------------


def test_non_finite_metric_is_diverged_and_has_no_plottable_value():
    """NaN and infinities are results [MML p.381], and have no position."""
    for raw in (float('nan'), float('inf'), float('-inf')):
        cell = reports.metric_cell(raw)
        assert cell['diverged'] is True
        assert cell['value'] is None
        assert cell['text'].startswith('diverged')


def test_huge_metric_is_diverged_so_it_cannot_swallow_an_axis():
    """4.9e+65 on an axis flattens every other point to one line."""
    cell = reports.metric_cell(4.896069537817991e65)
    assert cell['diverged'] is True
    assert cell['value'] is None
    assert '4.90e+65' in cell['text']
    assert reports.metric_cell(reports.DIVERGENCE_LIMIT)['diverged'] is False


def test_metric_cell_distinguishes_absent_from_unusable():
    """'not measured' and 'measured badly' are different facts."""
    absent = reports.metric_cell(None)
    assert absent['present'] is False
    assert absent['diverged'] is False
    assert absent['text'] == '--'
    odd = reports.metric_cell({'rmse': 1})
    assert odd['present'] is True
    assert odd['value'] is None
    assert 'not numeric' in odd['text']
    assert reports.metric_cell(True)['value'] is None


def test_metric_cell_recovers_a_number_that_lost_its_type():
    """json.dumps(default=str) can stringify a value on the way to disk."""
    assert reports.metric_cell('0.25')['value'] == pytest.approx(0.25)
    assert reports.metric_cell('nan')['diverged'] is True


def test_ci_cell_reports_a_malformed_interval_as_malformed():
    """Half an interval is not an interval."""
    assert reports.ci_cell([0.24, 0.27])['text'] == '[0.24, 0.27]'
    assert reports.ci_cell(None)['present'] is False
    assert 'malformed' in reports.ci_cell([0.24])['text']
    assert reports.ci_cell([float('nan'), 1.0])['diverged'] is True


def test_payload_is_json_safe_even_when_a_run_diverged(tmp_path):
    """The browser's JSON.parse rejects NaN; Python's encoder writes it."""
    payload = make_report(status=STATUS_PENDING)
    payload['arms'][1]['metrics']['rmse'] = float('nan')
    payload['arms'][1]['status'] = STATUS_PENDING
    raw = json.dumps(payload)
    assert 'NaN' in raw
    write_run(tmp_path, '20260118T093000-exp01_baseline', None, raw=raw)
    built = reports.build_payload(tmp_path)
    json.dumps(built, allow_nan=False)
    (run,) = built['runs']
    diverged = run['arms'][1]['metrics']['rmse']
    assert diverged['diverged'] is True
    assert diverged['value'] is None


# ---- configs, comparison and sweeps ----------------------------------


def test_flatten_config_uses_the_dotted_keys_the_cli_takes():
    """What the panel shows can be pasted straight back into --set."""
    flat = reports.flatten_config(make_report()['config'])
    assert flat['arms.gradient_descent.lr'] == 0.05
    assert flat['dataset.noise'] == 0.25
    assert flat['metric.secondary'] == ['mae']


def test_flatten_config_keeps_an_empty_arm_table():
    """`[arms.normal_equations]` declares an arm; losing it hides a diff."""
    flat = reports.flatten_config(make_report()['config'])
    assert flat['arms.normal_equations'] == {}
    assert reports.flatten_config('not a table') == {}


def test_overrides_are_excluded_from_the_config_comparison(tmp_path):
    """`overrides` records what changed; it is not itself a setting."""
    payload = make_report()
    payload['config']['overrides'] = ['arms.gradient_descent.lr=0.05']
    write_run(tmp_path, '20260118T093000-exp01_baseline', payload)
    (run,) = reports.load_runs(tmp_path)
    assert 'overrides' not in run['config_flat']
    assert run['overrides'] == ['arms.gradient_descent.lr=0.05']


def _sweep_fixture(tmp_path, values, key='arms.gradient_descent.lr'):
    """Writes one run per value, varying only `key`."""
    for index, value in enumerate(values):
        payload = make_report()
        node = payload['config']
        parts = key.split('.')
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
        payload['arms'][1]['params']['lr'] = value
        write_run(tmp_path, f'2026011{index}T093000-exp01_baseline', payload)


def test_sweep_detects_the_single_varying_key(tmp_path):
    """One key moved, every other held equal: that is a sweep."""
    _sweep_fixture(tmp_path, [0.01, 0.05, 0.5])
    groups = reports.sweep_groups(reports.load_runs(tmp_path))
    assert len(groups) == 1
    group = groups[0]
    assert group['key'] == 'arms.gradient_descent.lr'
    assert group['arm'] == 'gradient_descent'
    assert group['metric'] == 'rmse'
    assert [point['axis_text'] for point in group['points']] == [
        '0.01',
        '0.05',
        '0.5',
    ]
    assert all(
        point['arm_status'] == STATUS_VERIFIED for point in group['points']
    )


def test_sweep_counts_and_keeps_diverged_points(tmp_path):
    """A diverged point stays on the list with no value to plot.

    This is the lr sweep the program is built to produce: descent above
    the stability limit diverges [MML p.381], and the point has to stay
    visible without being given a place on the axis.
    """
    stable = make_report()
    stable['config']['arms']['gradient_descent']['lr'] = 0.05
    blown = make_report(status=STATUS_PENDING)
    blown['config']['arms']['gradient_descent']['lr'] = 1.1
    blown['arms'][1]['metrics'] = {'rmse': float('nan')}
    blown['arms'][1]['ci'] = {}
    blown['arms'][1]['status'] = STATUS_PENDING
    write_run(tmp_path, '20260118T093000-exp01_baseline', stable)
    write_run(tmp_path, '20260118T093001-exp01_baseline', blown)

    (group,) = reports.sweep_groups(reports.load_runs(tmp_path))
    assert group['diverged'] == 1
    assert [point['axis_text'] for point in group['points']] == ['0.05', '1.1']
    head, tail = group['points']
    assert head['metric']['value'] == pytest.approx(0.2563724945)
    assert tail['metric']['value'] is None
    assert tail['metric']['text'] == 'diverged (nan)'
    assert tail['arm_status'] == STATUS_PENDING


def test_runs_differing_in_two_keys_form_no_sweep(tmp_path):
    """Two moved keys explain nothing, so there is nothing to plot."""
    first = make_report()
    first['config']['arms']['gradient_descent']['lr'] = 0.05
    second = make_report()
    second['config']['arms']['gradient_descent']['lr'] = 0.5
    second['config']['dataset']['noise'] = 0.9
    write_run(tmp_path, '20260118T093000-exp01_baseline', first)
    write_run(tmp_path, '20260118T093001-exp01_baseline', second)
    assert reports.sweep_groups(reports.load_runs(tmp_path)) == []


def test_identical_runs_form_no_sweep(tmp_path):
    """Repeating a run is not a parameter study."""
    write_run(tmp_path, '20260118T093000-exp01_baseline', make_report())
    write_run(tmp_path, '20260118T093001-exp01_baseline', make_report())
    assert reports.sweep_groups(reports.load_runs(tmp_path)) == []


def test_sweeps_are_per_experiment(tmp_path):
    """Two experiments are not comparable, whatever their configs share."""
    first = make_report()
    first['config']['arms']['gradient_descent']['lr'] = 0.05
    second = make_report(experiment='exp02_tabular')
    second['config']['experiment'] = 'exp02_tabular'
    second['config']['arms']['gradient_descent']['lr'] = 0.5
    write_run(tmp_path, '20260118T093000-exp01_baseline', first)
    write_run(tmp_path, '20260118T093001-exp02_tabular', second)
    assert reports.sweep_groups(reports.load_runs(tmp_path)) == []


def test_unreadable_runs_never_enter_a_sweep(tmp_path):
    """A run with no config cannot hold any other key equal."""
    _sweep_fixture(tmp_path, [0.05, 0.5])
    write_run(tmp_path, '20260119T093000-exp01_baseline', None, raw='{')
    (group,) = reports.sweep_groups(reports.load_runs(tmp_path))
    assert len(group['points']) == 2


# ---- the banner ------------------------------------------------------


def test_banner_lists_every_run_that_is_not_verified(tmp_path):
    """Unverified results stay in view; nothing filters them out."""
    write_run(tmp_path, '20260118T093000-exp01_baseline', make_report())
    write_run(
        tmp_path,
        '20260118T093001-exp01_baseline',
        make_report(status=STATUS_PENDING),
    )
    write_run(
        tmp_path,
        '20260118T093002-exp01_baseline',
        make_report(status=STATUS_UNVERIFIED),
    )
    payload = reports.build_payload(tmp_path)
    assert payload['unverified'] == [
        '20260118T093002-exp01_baseline',
        '20260118T093001-exp01_baseline',
    ]
    assert len(payload['runs']) == 3


def test_runs_are_listed_newest_first(tmp_path):
    """The board opens on what was just run."""
    for stamp in ('20260118T093000', '20260118T093500', '20260117T120000'):
        write_run(tmp_path, f'{stamp}-exp01_baseline', make_report())
    runs = reports.load_runs(tmp_path)
    assert [run['run_id'].split('-')[0] for run in runs] == [
        '20260118T093500',
        '20260118T093000',
        '20260117T120000',
    ]
