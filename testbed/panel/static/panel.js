/*
 * Control panel renderer. Vanilla DOM, no dependencies, no build step.
 *
 * One invariant runs through the whole file: every number reaches the
 * page through `metricWithTag`, which cannot render a value without the
 * status tag of the arm that produced it. There is deliberately no
 * function that returns a bare formatted metric, because the moment one
 * exists, some table starts using it.
 *
 * Nothing here writes. There is no endpoint to write to.
 */

'use strict';

const API = '/api/runs.json';

const TAG_CLASS = {
  'VERIFIED': 'tag-verified',
  'PENDING REVISION': 'tag-pending',
  'UNVERIFIED': 'tag-unver',
  'BUG': 'tag-bug',
};

const POINT_CLASS = {
  'VERIFIED': 'pt-verified',
  'PENDING REVISION': 'pt-pending',
  'UNVERIFIED': 'pt-unver',
  'BUG': 'pt-bug',
};

const NO_TAG = 'NO TAG';
const UNREADABLE = 'UNREADABLE';

const state = {
  payload: null,
  tab: 'board',
  runId: null,
  experiment: null,
  picked: new Set(),
  showAllKeys: false,
};

/* ---- tiny DOM helpers ---------------------------------------------- */

function el(tag, props, kids) {
  const node = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === null || value === undefined) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key === 'dataset') Object.assign(node.dataset, value);
      else if (key.startsWith('on')) {
        node.addEventListener(key.slice(2).toLowerCase(), value);
      } else node.setAttribute(key, value);
    }
  }
  for (const kid of kids || []) {
    if (kid === null || kid === undefined) continue;
    node.append(typeof kid === 'string' ? document.createTextNode(kid) : kid);
  }
  return node;
}

function svgEl(name, attrs, kids) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined) continue;
    node.setAttribute(key, String(value));
  }
  for (const kid of kids || []) if (kid) node.append(kid);
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

/* ---- the tag chokepoint -------------------------------------------- */

/*
 * A status tag. `known` false means the report carried no tag, or one
 * the panel does not recognise; that is rendered hatched, because it is
 * a defect in the report and not a fifth verdict.
 */
function tagBadge(status, known) {
  if (!status) {
    return el('span', {
      class: 'tag tag-notag',
      text: NO_TAG,
      title: 'the report carried no status tag for this arm',
    });
  }
  if (!known) {
    return el('span', {
      class: 'tag tag-notag',
      text: status,
      title: 'not one of the four status tags this program defines',
    });
  }
  return el('span', {class: 'tag ' + TAG_CLASS[status], text: status});
}

/*
 * The only way a metric reaches the page. `cell` is what
 * panel.reports.metric_cell produced, so a diverged value arrives
 * labelled and with no number to plot.
 */
function metricWithTag(cell, status, known) {
  const safe = cell || {text: '--', diverged: false, present: false};
  let cls = 'value';
  if (safe.diverged) cls += ' diverged';
  else if (!safe.present) cls += ' absent';
  return el('span', {class: 'metric'}, [
    el('span', {class: cls, text: safe.text}),
    tagBadge(status, known),
  ]);
}

function statusBadgeForRun(run) {
  if (!run.readable) {
    return el('span', {
      class: 'tag tag-notag',
      text: UNREADABLE,
      title: 'report.json could not be parsed; this is not a status tag',
    });
  }
  return tagBadge(run.status, run.known_status);
}

/* ---- formatting ---------------------------------------------------- */

function fmtConfigValue(value) {
  if (value === null || value === undefined) return '--';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function fmtTimestamp(run) {
  if (!run.timestamp) return '(unknown)';
  const suffix = run.timestamp_source === 'directory' ? ' *' : '';
  return run.timestamp.replace('T', ' ') + suffix;
}

function fmtSeconds(value) {
  if (value === null || value === undefined) return '--';
  return value.toFixed(3);
}

/* The y-scale, the collapse decision and the tick labels arrive on
 * `group.axis` from reports.axis_scale. They are claims about the
 * numbers, not about pixels, so they are decided and tested in Python;
 * this file only maps them to coordinates. */

function grid(headers, rows) {
  const head = el('tr', null, headers.map((h) => el('th', {text: h})));
  return el('table', {class: 'grid'}, [
    el('thead', null, [head]),
    el('tbody', null, rows),
  ]);
}

function emptyNote(text) {
  return el('p', {class: 'empty', text: text});
}

function findRun(runId) {
  return (state.payload.runs || []).find((r) => r.run_id === runId) || null;
}

/* ---- banner -------------------------------------------------------- */

/*
 * Lists every run that is not VERIFIED. It is rendered before any tab,
 * it is never collapsible, and it is never filtered: a result nobody
 * can see is the failure this program is built around.
 */
function renderBanner() {
  const node = document.getElementById('banner');
  clear(node);
  const ids = state.payload.unverified || [];
  if (!ids.length) {
    node.hidden = true;
    return;
  }
  node.hidden = false;
  node.append(
    el('h2', {text: 'Not verified — ' + ids.length + ' run(s)'}),
    el('p', {
      text:
        'These runs stay listed here until a gate passes. An unverified ' +
        'number is not a provisional result; it is not a result.',
    }),
    el(
      'ul',
      null,
      ids.map((id) => {
        const run = findRun(id);
        return el('li', null, [
          el('label', {class: 'pick'}, [
            el('button', {
              type: 'button',
              text: id,
              onClick: () => openRun(id),
            }),
            run ? statusBadgeForRun(run) : tagBadge(null, false),
          ]),
        ]);
      }),
    ),
  );
}

function renderNotices() {
  const node = document.getElementById('notices');
  clear(node);
  const notices = state.payload.notices || [];
  if (!notices.length) {
    node.hidden = true;
    return;
  }
  node.hidden = false;
  node.append(
    el('strong', {text: 'Panel notices'}),
    el('ul', null, notices.map((t) => el('li', {text: t}))),
  );
}

/* ---- status board -------------------------------------------------- */

function renderBoard() {
  const host = clear(document.getElementById('panel-board'));
  const runs = state.payload.runs || [];
  const card = el('div', {class: 'card'}, [
    el('h2', {text: 'Status board'}),
    el('p', {
      class: 'hint',
      text:
        'One row per audit pack under runs/. No metric appears here: the ' +
        'board answers "what was run and is it trusted", and a number ' +
        'belongs next to the tag of the arm that produced it.',
    }),
  ]);
  if (!runs.length) {
    card.append(emptyNote('No runs found. Run an experiment first.'));
    host.append(card);
    return;
  }
  const rows = runs.map((run) =>
    el(
      'tr',
      {
        class: 'row-link',
        onClick: () => openRun(run.run_id),
        title: 'open this run',
      },
      [
        el('td', {class: 'mono', text: run.experiment}),
        el('td', {class: 'mono', text: fmtTimestamp(run)}),
        el('td', {
          class: 'mono num',
          text: run.seed === null ? '--' : String(run.seed),
        }),
        el('td', {class: 'mono', text: run.dataset_checksum || '--'}),
        el('td', null, [statusBadgeForRun(run)]),
        el('td', {
          class: run.errors.length ? 'errors' : 'notes',
          text: run.errors.length
            ? run.errors.length + ' parse problem(s)'
            : '',
        }),
        el('td', {class: 'mono notes', text: run.run_id}),
      ],
    ),
  );
  card.append(
    grid(
      [
        'experiment',
        'timestamp',
        'seed',
        'dataset checksum',
        'status',
        'parse',
        'run id',
      ],
      rows,
    ),
  );
  card.append(
    el('p', {
      class: 'hint',
      text:
        '* timestamp recovered from the directory name: this report was ' +
        'written before report.json carried created_at.',
    }),
  );
  host.append(card);
}

/* ---- run detail ---------------------------------------------------- */

function runPicker(onChange, selected) {
  const options = (state.payload.runs || []).map((run) =>
    el('option', {
      value: run.run_id,
      text:
        run.run_id +
        '  —  ' +
        (run.readable ? run.status || NO_TAG : UNREADABLE),
      selected: run.run_id === selected ? 'selected' : null,
    }),
  );
  return el('select', {onChange: (ev) => onChange(ev.target.value)}, options);
}

function armRows(run) {
  const rows = [];
  for (const arm of run.arms) {
    const names = Object.keys(arm.metrics);
    const tag = () => metricWithTag(null, arm.status, arm.known_status);
    if (!names.length) {
      rows.push(
        el('tr', {class: arm.present ? '' : 'absent'}, [
          el('td', {class: 'mono', text: arm.name}),
          el('td', {class: 'mono', text: '--'}),
          el('td', null, [tag()]),
          el('td', {class: 'mono', text: '--'}),
          el('td', {class: 'mono num', text: fmtSeconds(arm.seconds)}),
        ]),
      );
      continue;
    }
    const primaryFirst = names.slice().sort((a, b) => {
      if (a === run.primary_metric) return -1;
      if (b === run.primary_metric) return 1;
      return a.localeCompare(b);
    });
    primaryFirst.forEach((name, index) => {
      const ci = arm.ci[name];
      rows.push(
        el('tr', null, [
          el('td', {class: 'mono', text: index === 0 ? arm.name : ''}),
          el('td', {
            class: 'mono',
            text: name + (name === run.primary_metric ? '  (primary)' : ''),
          }),
          el('td', null, [
            metricWithTag(arm.metrics[name], arm.status, arm.known_status),
          ]),
          el('td', {class: 'mono', text: ci ? ci.text : '--'}),
          el('td', {
            class: 'mono num',
            text: index === 0 ? fmtSeconds(arm.seconds) : '',
          }),
        ]),
      );
    });
  }
  return rows;
}

function renderRun() {
  const host = clear(document.getElementById('panel-run'));
  const runs = state.payload.runs || [];
  if (!runs.length) {
    host.append(
      el('div', {class: 'card'}, [
        emptyNote('No runs found. Run an experiment first.'),
      ]),
    );
    return;
  }
  if (!state.runId || !findRun(state.runId)) state.runId = runs[0].run_id;
  const run = findRun(state.runId);

  const head = el('div', {class: 'card'}, [
    el('h2', {text: 'Run detail'}),
    el('div', {class: 'pill-row'}, [
      el('span', {text: 'run:'}),
      runPicker((id) => {
        state.runId = id;
        renderRun();
      }, state.runId),
      statusBadgeForRun(run),
    ]),
    grid(
      ['field', 'value'],
      [
        ['experiment', run.experiment],
        ['run id', run.run_id],
        ['timestamp', fmtTimestamp(run) + ' (' + run.timestamp_source + ')'],
        ['seed', run.seed === null ? '--' : String(run.seed)],
        ['dataset', run.dataset_card.name || '--'],
        ['dataset checksum', run.dataset_checksum || '--'],
        ['primary metric', run.primary_metric || '(none declared)'],
        [
          'overrides',
          run.overrides.length ? run.overrides.join('  ') : '(none)',
        ],
      ].map(([k, v]) =>
        el('tr', null, [
          el('td', {text: k}),
          el('td', {class: 'mono', text: v}),
        ]),
      ),
    ),
  ]);
  host.append(head);

  if (run.errors.length) {
    host.append(
      el('div', {class: 'card'}, [
        el('h2', {text: 'Parse problems'}),
        el('p', {
          class: 'hint',
          text:
            'The panel reports what the report does not carry rather than ' +
            'filling it in. Nothing below was inferred.',
        }),
        el(
          'ul',
          {class: 'lines errors'},
          run.errors.map((t) => el('li', {text: t})),
        ),
      ]),
    );
  }

  if (!run.readable) return;

  host.append(
    el('div', {class: 'card'}, [
      el('h2', {text: 'Arms'}),
      el('p', {
        class: 'hint',
        text:
          'Every value carries the tag of the arm that produced it. ' +
          'Seconds are wall-clock on this machine and have no source ' +
          'baseline (GAP-12).',
      }),
      grid(['arm', 'metric', 'value', '95% CI', 'seconds'], armRows(run)),
    ]),
  );

  const notes = run.arms.filter((arm) => arm.notes);
  if (notes.length) {
    host.append(
      el('div', {class: 'card'}, [
        el('h2', {text: 'Arm notes'}),
        grid(
          ['arm', 'status', 'note'],
          notes.map((arm) =>
            el('tr', null, [
              el('td', {class: 'mono', text: arm.name}),
              el('td', null, [tagBadge(arm.status, arm.known_status)]),
              el('td', {text: arm.notes}),
            ]),
          ),
        ),
      ]),
    );
  }

  host.append(
    el('div', {class: 'card'}, [
      el('h2', {text: 'Gates'}),
      el('p', {
        class: 'hint',
        text:
          'Declared before the code ran, with the page the threshold came ' +
          'from. A gate with no anchor is not a gate.',
      }),
      run.gates.length
        ? grid(
            ['gate', 'page anchor', 'declared threshold', 'observed', ''],
            run.gates.map((gate) =>
              el('tr', null, [
                el('td', {class: 'mono', text: gate.name}),
                el('td', {text: gate.anchor || '(no anchor)'}),
                el('td', {class: 'mono', text: gate.threshold || '--'}),
                el('td', {class: 'mono', text: gate.observed || '--'}),
                el('td', {
                  class: gate.passed === true ? 'pass' : 'fail',
                  text:
                    gate.passed === true
                      ? 'PASS'
                      : gate.passed === false
                        ? 'FAIL'
                        : 'NOT STATED',
                }),
              ]),
            ),
          )
        : emptyNote('No gates in this report, so there is nothing to judge '
            + 'the run by.'),
    ]),
  );

  host.append(
    el('div', {class: 'card'}, [
      el('h2', {text: 'Split boundaries'}),
      run.splits.length
        ? el(
            'ul',
            {class: 'lines'},
            run.splits.map((t) => el('li', {text: t})),
          )
        : emptyNote('No split lines recorded.'),
      el('h3', {text: 'Known gaps affecting this run'}),
      run.gaps.length
        ? el('ul', {class: 'lines'}, run.gaps.map((t) => el('li', {text: t})))
        : emptyNote('None recorded.'),
    ]),
  );

  const flat = run.config_flat;
  host.append(
    el('div', {class: 'card'}, [
      el('h2', {text: 'Config'}),
      el('p', {
        class: 'hint',
        text:
          'Dotted keys are the ones --set and --sweep take, so anything ' +
          'here can be pasted back into a command.',
      }),
      grid(
        ['key', 'value'],
        Object.keys(flat)
          .sort()
          .map((key) =>
            el('tr', null, [
              el('td', {class: 'mono', text: key}),
              el('td', {class: 'mono', text: fmtConfigValue(flat[key])}),
            ]),
          ),
      ),
    ]),
  );

  const env = run.env || {};
  const packages = env.packages || {};
  host.append(
    el('div', {class: 'card'}, [
      el('h2', {text: 'Environment'}),
      grid(
        ['field', 'value'],
        [
          ['python', env.python || '--'],
          ['executable', env.executable || '--'],
          ['platform', env.platform || '--'],
        ]
          .map(([k, v]) =>
            el('tr', null, [
              el('td', {text: k}),
              el('td', {class: 'mono', text: v}),
            ]),
          )
          .concat(
            Object.keys(packages)
              .sort()
              .map((name) =>
                el('tr', null, [
                  el('td', {text: name}),
                  el('td', {
                    class: packages[name] === 'MISSING' ? 'mono errors' : 'mono',
                    text: String(packages[name]),
                  }),
                ]),
              ),
          ),
      ),
      el('h3', {text: 'Dataset card'}),
      el('pre', {
        class: 'block',
        text: JSON.stringify(run.dataset_card, null, 2),
      }),
    ]),
  );
}

function openRun(runId) {
  state.runId = runId;
  selectTab('run');
}

/* ---- compare ------------------------------------------------------- */

function experimentsWithRuns() {
  const groups = new Map();
  for (const run of state.payload.runs || []) {
    if (!run.readable) continue;
    if (!groups.has(run.experiment)) groups.set(run.experiment, []);
    groups.get(run.experiment).push(run);
  }
  return groups;
}

function renderCompare() {
  const host = clear(document.getElementById('panel-compare'));
  const groups = experimentsWithRuns();
  if (!groups.size) {
    host.append(
      el('div', {class: 'card'}, [
        emptyNote('No readable runs to compare.'),
      ]),
    );
    return;
  }
  const names = [...groups.keys()].sort();
  if (!state.experiment || !groups.has(state.experiment)) {
    state.experiment = names.reduce((best, name) =>
      groups.get(name).length > groups.get(best).length ? name : best,
    names[0]);
    state.picked = new Set();
  }
  const members = groups.get(state.experiment);
  if (!state.picked.size) {
    state.picked = new Set(members.slice(0, 2).map((r) => r.run_id));
  }
  const picked = members.filter((r) => state.picked.has(r.run_id));

  const controls = el('div', {class: 'card'}, [
    el('h2', {text: 'Compare runs of one experiment'}),
    el('p', {
      class: 'hint',
      text:
        'Only runs of the same experiment are comparable: the arms, the ' +
        'gates and the metric all have to mean the same thing.',
    }),
    el('div', {class: 'pill-row'}, [
      el('span', {text: 'experiment:'}),
      el(
        'select',
        {
          onChange: (ev) => {
            state.experiment = ev.target.value;
            state.picked = new Set();
            renderCompare();
          },
        },
        names.map((name) =>
          el('option', {
            value: name,
            text: name + ' (' + groups.get(name).length + ' runs)',
            selected: name === state.experiment ? 'selected' : null,
          }),
        ),
      ),
    ]),
    el(
      'div',
      {class: 'pill-row'},
      members.map((run) =>
        el('label', {class: 'pick'}, [
          el('input', {
            type: 'checkbox',
            checked: state.picked.has(run.run_id) ? 'checked' : null,
            onChange: (ev) => {
              if (ev.target.checked) state.picked.add(run.run_id);
              else state.picked.delete(run.run_id);
              renderCompare();
            },
          }),
          el('span', {class: 'mono', text: run.run_id.slice(0, 15)}),
          statusBadgeForRun(run),
        ]),
      ),
    ),
  ]);
  host.append(controls);

  if (picked.length < 2) {
    host.append(
      el('div', {class: 'card'}, [
        emptyNote('Pick at least two runs to diff.'),
      ]),
    );
    return;
  }

  const keys = [...new Set(picked.flatMap((r) => Object.keys(r.config_flat)))]
    .sort();
  const differing = keys.filter((key) => {
    const seen = new Set(
      picked.map((r) => JSON.stringify(r.config_flat[key] ?? null)),
    );
    return seen.size > 1;
  });
  const shown = state.showAllKeys ? keys : differing;

  host.append(
    el('div', {class: 'card'}, [
      el('h2', {text: 'Config diff'}),
      el('p', {
        class: 'hint',
        text:
          differing.length +
          ' of ' +
          keys.length +
          ' config keys differ across the picked runs.',
      }),
      el('div', {class: 'pill-row'}, [
        el('button', {
          type: 'button',
          text: state.showAllKeys ? 'Show only differences' : 'Show every key',
          onClick: () => {
            state.showAllKeys = !state.showAllKeys;
            renderCompare();
          },
        }),
      ]),
      shown.length
        ? grid(
            ['config key'].concat(picked.map((r) => r.run_id.slice(0, 15))),
            shown.map((key) =>
              el(
                'tr',
                {class: differing.includes(key) ? 'changed' : ''},
                [el('td', {class: 'mono', text: key})].concat(
                  picked.map((r) =>
                    el('td', {
                      class: 'mono',
                      text: fmtConfigValue(r.config_flat[key]),
                    }),
                  ),
                ),
              ),
            ),
          )
        : emptyNote('These runs share every config value.'),
    ]),
  );

  const armNames = [
    ...new Set(picked.flatMap((r) => r.arms.map((a) => a.name))),
  ];
  const metricName = picked[0].primary_metric;
  const rows = armNames.map((name) => {
    const cells = [el('td', {class: 'mono', text: name})];
    for (const run of picked) {
      const arm = run.arms.find((a) => a.name === name);
      const cell = arm ? arm.metrics[run.primary_metric] : null;
      cells.push(
        el('td', null, [
          arm
            ? metricWithTag(cell, arm.status, arm.known_status)
            : metricWithTag(null, null, false),
        ]),
      );
    }
    if (picked.length === 2) {
      const [a, b] = picked.map((run) => {
        const arm = run.arms.find((x) => x.name === name);
        const cell = arm ? arm.metrics[run.primary_metric] : null;
        return cell && cell.value !== null ? cell.value : null;
      });
      let text = 'n/a';
      if (a !== null && b !== null) {
        const delta = b - a;
        const pct = a !== 0 ? (delta / Math.abs(a)) * 100 : NaN;
        text =
          (delta >= 0 ? '+' : '') +
          delta.toPrecision(4) +
          (Number.isFinite(pct)
            ? '  (' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)'
            : '');
      } else text = 'n/a (a value diverged or is absent)';
      cells.push(el('td', {class: 'mono num', text: text}));
    }
    return el('tr', null, cells);
  });

  const headers = ['arm'].concat(picked.map((r) => r.run_id.slice(0, 15)));
  if (picked.length === 2) headers.push('delta');
  host.append(
    el('div', {class: 'card'}, [
      el('h2', {
        text: 'Primary metric: ' + (metricName || '(none declared)'),
      }),
      el('p', {
        class: 'hint',
        text:
          'Read this next to the diff above: with one config key changed, ' +
          'that key is the only candidate explanation for a moved metric.',
      }),
      grid(headers, rows),
    ]),
  );
}

/* ---- sweeps -------------------------------------------------------- */

function sweepPlot(group) {
  const W = 760;
  const H = 330;
  const left = 70;
  const right = W - 24;
  const bandTop = 10;
  const bandBottom = 34;
  const plotTop = 54;
  const plotBottom = H - 52;

  const points = group.points;
  const finite = points.filter((p) => p.metric.value !== null);
  const diverged = points.filter((p) => p.metric.value === null);
  const numeric = points.every((p) => p.axis_value !== null);

  const xOf = (point, index) => {
    if (!numeric) {
      const step = (right - left) / Math.max(points.length, 1);
      return left + step * (index + 0.5);
    }
    const values = points.map((p) => p.axis_value);
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    if (hi === lo) return (left + right) / 2;
    return left + ((point.axis_value - lo) / (hi - lo)) * (right - left);
  };

  const axis = group.axis || {collapsed: false, lo: 0, hi: 1, ticks: []};
  const collapsed = Boolean(axis.collapsed);
  const commonValue = axis.common_value;
  const yLo = axis.lo;
  const yHi = axis.hi;
  const yOf = (value) =>
    plotBottom - ((value - yLo) / (yHi - yLo)) * (plotBottom - plotTop);

  const kids = [];

  /* The diverged band is its own strip, off the axis entirely, and is
   * drawn only when something actually diverged. */
  if (diverged.length) {
    kids.push(
      svgEl('rect', {
        class: 'diverged-band',
        x: left,
        y: bandTop,
        width: right - left,
        height: bandBottom - bandTop,
      }),
      svgEl('line', {
        class: 'diverged-edge',
        x1: left,
        y1: bandBottom + 6,
        x2: right,
        y2: bandBottom + 6,
      }),
      svgEl('text', {
        class: 'diverged-label',
        x: left + 6,
        y: bandTop + 15,
      }, [document.createTextNode(
        'diverged — off scale, never plotted on the axis below',
      )]),
    );
  }

  if (finite.length && collapsed) {
    /* One line, one label, and a note saying why: the points really are
     * the same value, and a chart that implied otherwise would be the
     * mathematically-wrong-but-pretty picture this testbed guards against. */
    const tick = axis.ticks[0];
    const y = yOf(commonValue);
    kids.push(
      svgEl('line', {
        class: 'grid-line',
        x1: left,
        y1: y,
        x2: right,
        y2: y,
      }),
      svgEl('text', {class: 'tick', x: left - 8, y: y + 3,
        'text-anchor': 'end'},
      [document.createTextNode(tick ? tick.label : String(commonValue))]),
      svgEl('text', {
        class: 'axis-label',
        x: (left + right) / 2,
        y: y - 12,
        'text-anchor': 'middle',
      }, [
        svgEl('title', null, [document.createTextNode(
          'all ' + finite.length + ' points equal ' + commonValue,
        )]),
        document.createTextNode(
          'axis collapsed — all ' + finite.length +
          ' points are the same value',
        ),
      ]),
    );
  } else if (finite.length) {
    axis.ticks.forEach((tick) => {
      const value = tick.value;
      const y = yOf(value);
      kids.push(
        svgEl('line', {
          class: 'grid-line',
          x1: left,
          y1: y,
          x2: right,
          y2: y,
        }),
        svgEl('text', {class: 'tick', x: left - 8, y: y + 3,
          'text-anchor': 'end'},
        [document.createTextNode(tick.label)]),
      );
    });
  } else {
    kids.push(
      svgEl('text', {
        class: 'diverged-label',
        x: (left + right) / 2,
        y: (plotTop + plotBottom) / 2,
        'text-anchor': 'middle',
      }, [document.createTextNode(
        'every point diverged — there is no scale to draw',
      )]),
    );
  }

  kids.push(
    svgEl('line', {
      class: 'axis',
      x1: left,
      y1: plotBottom,
      x2: right,
      y2: plotBottom,
    }),
    svgEl('line', {
      class: 'axis',
      x1: left,
      y1: plotTop,
      x2: left,
      y2: plotBottom,
    }),
  );

  /* The line breaks at a diverged point: there is no continuity to draw
   * through a value that left the scale. */
  let segment = [];
  const flushSegment = () => {
    if (segment.length > 1) {
      kids.push(
        svgEl('polyline', {
          class: 'series',
          points: segment.map(([x, y]) => x + ',' + y).join(' '),
        }),
      );
    }
    segment = [];
  };
  points.forEach((point, index) => {
    if (point.metric.value === null) {
      flushSegment();
      return;
    }
    segment.push([xOf(point, index), yOf(point.metric.value)]);
  });
  flushSegment();

  points.forEach((point, index) => {
    const x = xOf(point, index);
    const cls =
      POINT_CLASS[point.arm_status] && point.known_status
        ? POINT_CLASS[point.arm_status]
        : 'pt-notag';
    const label =
      point.run_id +
      '\n' +
      group.key +
      ' = ' +
      point.axis_text +
      '\n' +
      (group.metric || 'metric') +
      ' = ' +
      point.metric.text +
      '\n' +
      'arm ' +
      (point.arm || '?') +
      ': ' +
      (point.arm_status || NO_TAG);
    const title = svgEl('title', null, [document.createTextNode(label)]);
    if (point.metric.value === null) {
      kids.push(
        svgEl('rect', {
          class: cls,
          x: x - 5,
          y: (bandTop + bandBottom) / 2 - 5,
          width: 10,
          height: 10,
          transform: 'rotate(45 ' + x + ' ' + (bandTop + bandBottom) / 2 + ')',
        }, [title]),
      );
    } else {
      kids.push(
        svgEl('circle', {
          class: cls,
          cx: x,
          cy: yOf(point.metric.value),
          r: 5,
        }, [title]),
      );
    }
    kids.push(
      svgEl('text', {
        class: 'tick',
        x: x,
        y: plotBottom + 16,
        'text-anchor': 'middle',
      }, [document.createTextNode(point.axis_text)]),
    );
  });

  kids.push(
    svgEl('text', {
      class: 'axis-label',
      x: (left + right) / 2,
      y: H - 10,
      'text-anchor': 'middle',
    }, [document.createTextNode(group.key)]),
    svgEl('text', {
      class: 'axis-label',
      x: 14,
      y: (plotTop + plotBottom) / 2,
      transform:
        'rotate(-90 14 ' + ((plotTop + plotBottom) / 2) + ')',
      'text-anchor': 'middle',
    }, [document.createTextNode(group.metric || 'primary metric')]),
  );

  const svg = svgEl(
    'svg',
    {
      class: 'plot',
      viewBox: '0 0 ' + W + ' ' + H,
      role: 'img',
      'aria-label':
        'primary metric against ' +
        group.key +
        ', ' +
        diverged.length +
        ' diverged point(s)',
    },
    kids,
  );
  return svg;
}

function renderSweep() {
  const host = clear(document.getElementById('panel-sweep'));
  const sweeps = state.payload.sweeps || [];
  host.append(
    el('div', {class: 'card'}, [
      el('h2', {text: 'Sweeps'}),
      el('p', {
        class: 'hint',
        text:
          'A sweep is a set of runs whose configs are identical except ' +
          'for one key. That is what makes the one key the only candidate ' +
          'explanation for a change in the metric. Runs differing in two ' +
          'keys form no sweep, because such a pair explains nothing.',
      }),
      sweeps.length
        ? el('p', {
            class: 'hint',
            text: sweeps.length + ' sweep(s) detected under runs/.',
          })
        : emptyNote(
            'No sweep detected. Generate one with repeated --set runs, ' +
              'e.g. --set arms.gradient_descent.lr=0.05 then 0.9 then 1.1.',
          ),
    ]),
  );

  for (const group of sweeps) {
    const rows = group.points.map((point) =>
      el('tr', null, [
        el('td', {class: 'mono num', text: point.axis_text}),
        el('td', null, [
          metricWithTag(point.metric, point.arm_status, point.known_status),
        ]),
        el('td', {class: 'mono', text: point.arm || '--'}),
        el('td', null, [
          point.run_status
            ? tagBadge(point.run_status, true)
            : tagBadge(null, false),
        ]),
        el('td', null, [
          el('button', {
            type: 'button',
            text: point.run_id.slice(0, 15),
            onClick: () => openRun(point.run_id),
          }),
        ]),
      ]),
    );
    host.append(
      el('div', {class: 'card'}, [
        el('h2', {text: group.key}),
        el('p', {
          class: 'hint',
          text:
            group.experiment +
            ' · arm ' +
            (group.arm || '(first reported)') +
            ' · metric ' +
            (group.metric || '(none declared)') +
            ' · ' +
            group.points.length +
            ' points, ' +
            group.diverged +
            ' diverged',
        }),
        sweepPlot(group),
        el('p', {
          class: 'hint',
          text:
            'Divergence above the stability limit is a required result of ' +
            'this sweep, not a failure: see [MML p.381] and [MML p.389].',
        }),
        grid(
          [group.key, group.metric || 'metric', 'arm', 'run status', 'run'],
          rows,
        ),
      ]),
    );
  }
}

/* ---- legend -------------------------------------------------------- */

function renderLegend() {
  const host = clear(document.getElementById('panel-legend'));
  const meanings = [
    [
      'VERIFIED',
      'Every gate passed and at least one arm produced a number the ' +
        'gates accepted.',
    ],
    [
      'PENDING REVISION',
      'A gate failed. The numbers exist and are not to be used.',
    ],
    [
      'UNVERIFIED',
      'There was nothing to judge the run by, or the arm could not run ' +
        '(a missing dependency, for instance). Not a weak pass.',
    ],
    ['BUG', 'The run itself is known to be wrong.'],
  ];
  host.append(
    el('div', {class: 'card'}, [
      el('h2', {text: 'The four tags'}),
      el('p', {
        class: 'hint',
        text:
          'Tags come from the run. The panel never derives, upgrades or ' +
          'supplies one.',
      }),
      grid(
        ['tag', 'what it means'],
        meanings.map(([tag, meaning]) =>
          el('tr', null, [
            el('td', null, [tagBadge(tag, true)]),
            el('td', {text: meaning}),
          ]),
        ),
      ),
      el('h3', {text: 'And two things that are not tags'}),
      grid(
        ['marker', 'what it means'],
        [
          el('tr', null, [
            el('td', null, [
              el('span', {class: 'tag tag-notag', text: UNREADABLE}),
            ]),
            el('td', {
              text:
                'report.json could not be parsed. A parse outcome, not a ' +
                'verdict: the panel does not get to decide a run status.',
            }),
          ]),
          el('tr', null, [
            el('td', null, [
              el('span', {class: 'tag tag-notag', text: NO_TAG}),
            ]),
            el('td', {
              text:
                'The report carried a value but no status for the arm that ' +
                'produced it. The number is still shown, loudly untagged, ' +
                'because hiding it from a reviewer is worse.',
            }),
          ]),
        ],
      ),
      el('h3', {text: 'Divergence'}),
      el('p', {
        text:
          'A metric that is non-finite, or larger in magnitude than ' +
          (state.payload.divergence_limit || 1e6) +
          ', is reported as diverged and is given no position on any ' +
          'axis. It is a result of this program, not a failure of it ' +
          '[MML p.381].',
      }),
    ]),
  );
}

/* ---- shell --------------------------------------------------------- */

const RENDERERS = {
  board: renderBoard,
  run: renderRun,
  compare: renderCompare,
  sweep: renderSweep,
  legend: renderLegend,
};

function selectTab(name) {
  state.tab = name;
  for (const button of document.querySelectorAll('#tabs button')) {
    button.setAttribute(
      'aria-selected',
      button.dataset.tab === name ? 'true' : 'false',
    );
  }
  for (const section of document.querySelectorAll('.tab-panel')) {
    section.hidden = section.id !== 'panel-' + name;
  }
  RENDERERS[name]();
}

function renderAll() {
  document.getElementById('source-line').textContent =
    'reading ' + state.payload.runs_root +
    '  ·  ' + (state.payload.runs || []).length + ' run(s)';
  document.getElementById('generated-line').textContent =
    'read at ' + (state.payload.generated_at || '');
  renderBanner();
  renderNotices();
  selectTab(state.tab);
}

async function load() {
  const line = document.getElementById('source-line');
  try {
    const response = await fetch(API, {cache: 'no-store'});
    if (!response.ok) throw new Error('HTTP ' + response.status);
    state.payload = await response.json();
  } catch (error) {
    line.textContent = 'could not read ' + API + ': ' + error.message;
    return;
  }
  renderAll();
}

document.getElementById('refresh').addEventListener('click', load);
for (const button of document.querySelectorAll('#tabs button')) {
  button.addEventListener('click', () => selectTab(button.dataset.tab));
}
load();
