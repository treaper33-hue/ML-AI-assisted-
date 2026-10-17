"""Contract tests for the sweep chart's y-axis.

The axis is where a chart can lie most cheaply. Two failures matter:

  * drawing a slope through points that are identical, which turns
    float noise into a trend the data does not contain, and
  * printing tick labels too coarse to tell the ticks apart, which
    claims a resolution the axis does not have.

The rule lives in `panel.reports.axis_scale` rather than in the browser
precisely so these can be tested: it is a claim about the numbers, and
the renderer only maps the result to coordinates.

The motivating case is real. Three EXP-01 runs at learning rates 0.01,
0.05 and 0.2 all converge to the same least-squares solution; their
RMSE values differ by 2.8e-16, which is float noise. The first version
of the chart drew a steep line through them.
"""

import math

import pytest
from test_panel_reports import make_report
from test_panel_reports import write_run

from panel import reports

#: The three RMSE values that motivated the collapse rule, taken from
#: real EXP-01 runs that differ only in learning rate.
NOISE_LEVEL_VALUES = (
    0.2563724945054188,
    0.2563724945054191,
    0.2563724945054191,
)


def test_identical_values_collapse_the_axis():
    """Points that are exactly equal get one flat line, not a slope."""
    scale = reports.axis_scale([0.5, 0.5, 0.5])
    assert scale['collapsed'] is True
    assert scale['common_value'] == pytest.approx(0.5)
    assert len(scale['ticks']) == 1


def test_float_noise_collapses_the_axis():
    """The real EXP-01 case: a 2.8e-16 spread is not a trend."""
    scale = reports.axis_scale(list(NOISE_LEVEL_VALUES))
    assert scale['collapsed'] is True
    assert scale['relative_spread'] < reports.COLLAPSE_RELATIVE_SPREAD
    assert scale['common_value'] == pytest.approx(max(NOISE_LEVEL_VALUES))


def test_collapsed_axis_brackets_the_common_value():
    """The shared value sits inside the drawing area, not on its edge."""
    scale = reports.axis_scale(list(NOISE_LEVEL_VALUES))
    assert scale['lo'] < scale['common_value'] < scale['hi']


def test_a_real_difference_does_not_collapse():
    """A spread the sweep is meant to reveal must survive."""
    scale = reports.axis_scale([0.1, 0.25, 0.5])
    assert scale['collapsed'] is False
    assert scale['common_value'] is None
    assert len(scale['ticks']) == reports.AXIS_TICK_INTERVALS + 1


def test_the_collapse_boundary_is_where_it_claims_to_be():
    """Just inside the threshold collapses; just outside does not."""
    base = 1.0
    inside = base * (1.0 + reports.COLLAPSE_RELATIVE_SPREAD / 10.0)
    outside = base * (1.0 + reports.COLLAPSE_RELATIVE_SPREAD * 10.0)
    assert reports.axis_scale([base, inside])['collapsed'] is True
    assert reports.axis_scale([base, outside])['collapsed'] is False


def test_tick_labels_are_always_distinct():
    """Two ticks printing the same text would be an axis lying."""
    for values in ([0.1, 0.25, 0.5], [1.0, 1.0000001], [-5.0, 5.0]):
        scale = reports.axis_scale(values)
        if scale['collapsed']:
            continue
        labels = [tick['label'] for tick in scale['ticks']]
        assert len(set(labels)) == len(labels), labels


def test_tick_precision_rises_only_as_far_as_needed():
    """Close values get more figures; separated values stay readable."""
    assert reports.tick_digits([0.1, 0.2, 0.3]) == reports.MIN_TICK_DIGITS
    close = [0.25637240, 0.25637241, 0.25637242]
    assert reports.tick_digits(close) > reports.MIN_TICK_DIGITS


def test_diverged_points_do_not_reach_the_scale():
    """A None value is off the axis and must not widen it [TSP p.144].

    This is the same rule as the diverged band: a value that left the
    scale cannot be allowed to set the scale.
    """
    with_none = reports.axis_scale([0.1, None, 0.5])
    without = reports.axis_scale([0.1, 0.5])
    assert with_none['lo'] == pytest.approx(without['lo'])
    assert with_none['hi'] == pytest.approx(without['hi'])


def test_all_points_diverged_yields_a_drawable_but_empty_scale():
    """No finite point means no ticks, and no crash."""
    scale = reports.axis_scale([None, None])
    assert scale['ticks'] == []
    assert scale['collapsed'] is False
    assert scale['lo'] < scale['hi']


def test_zero_values_do_not_divide_by_zero():
    """A sweep that is all zeros collapses instead of exploding."""
    scale = reports.axis_scale([0.0, 0.0])
    assert scale['collapsed'] is True
    assert scale['common_value'] == pytest.approx(0.0)
    assert math.isfinite(scale['lo']) and math.isfinite(scale['hi'])


def test_negative_values_are_handled():
    """Metrics such as r2 go negative; the axis must cope."""
    scale = reports.axis_scale([-3.0, -1.0, 0.5])
    assert scale['collapsed'] is False
    assert scale['lo'] < -3.0 < 0.5 < scale['hi']


def test_format_significant_keeps_trailing_zeros():
    """Tick columns line up only if trailing zeros survive."""
    assert reports.format_significant(0.562, 4) == '0.5620'
    assert reports.format_significant(0.05333333, 4) == '0.05333'
    assert reports.format_significant(0.1, 4) == '0.1000'


def test_format_significant_rejects_nonsense_precision():
    """Zero significant figures is not a format, it is a bug."""
    with pytest.raises(ValueError, match='digits must be'):
        reports.format_significant(1.0, 0)


def test_axis_scale_returns_the_whole_contract():
    """The renderer reads these keys by name and has no fallback."""
    scale = reports.axis_scale([1.0, 2.0])
    assert set(scale) == {
        'collapsed',
        'common_value',
        'relative_spread',
        'lo',
        'hi',
        'ticks',
    }


def _run_at(lr, rmse):
    """Builds a report that differs only in learning rate and RMSE."""
    report = make_report()
    report['config']['arms']['gradient_descent']['lr'] = lr
    for arm in report['arms']:
        if arm['name'] == 'gradient_descent':
            arm['metrics']['rmse'] = rmse
    return report


def test_a_real_sweep_reaches_the_browser_already_collapsed(tmp_path):
    """End to end: the decision travels in the payload, not the page.

    This is the EXP-01 learning-rate sweep in miniature. If the group
    arrived without `axis`, the browser would have no rule to apply and
    the false slope would come back.
    """
    write_run(
        tmp_path,
        '20260118T093000-exp01_baseline',
        _run_at(0.01, NOISE_LEVEL_VALUES[0]),
    )
    write_run(
        tmp_path,
        '20260118T093100-exp01_baseline',
        _run_at(0.2, NOISE_LEVEL_VALUES[1]),
    )

    payload = reports.build_payload(tmp_path)
    assert payload['collapse_relative_spread'] == (
        reports.COLLAPSE_RELATIVE_SPREAD
    )
    (group,) = payload['sweeps']
    assert group['key'] == 'arms.gradient_descent.lr'
    assert group['axis']['collapsed'] is True
    assert len(group['axis']['ticks']) == 1


def test_a_sweep_with_a_real_difference_keeps_its_ticks(tmp_path):
    """The same path, with values a sweep is actually meant to show."""
    write_run(tmp_path, '20260118T093000-exp01_baseline', _run_at(0.01, 0.9))
    write_run(tmp_path, '20260118T093100-exp01_baseline', _run_at(0.2, 0.2))

    (group,) = reports.build_payload(tmp_path)['sweeps']
    axis = group['axis']
    assert axis['collapsed'] is False
    labels = [tick['label'] for tick in axis['ticks']]
    assert len(set(labels)) == len(labels)
