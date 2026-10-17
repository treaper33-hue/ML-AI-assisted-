# Testbed

The runnable half of the learning program. One command brings it up from
scratch; after that every experiment shares one loader, one scorer and one
set of splitters, so comparing two methodologies compares the methodologies
and nothing else.

## Boot

The script resolves an interpreter, installs what is needed, prints what this
machine can verify, runs the contract tests, and executes the baseline
experiment. Requires CPython 3.11 or newer: configs are TOML, read with the
standard library's `tomllib`.

**Clean and pinned** — creates `.venv` and installs the exact versions in
`requirements.txt`. Use this when a result has to reproduce elsewhere.

```powershell
.\setup.ps1
```

```bash
./setup.sh
```

**An environment you already have** — installs only the packages that are
absent and leaves every installed version alone. Pins are *not* enforced, so
results reproduce on that machine rather than from a clean clone; say so in
the run report (ledger GAP-14).

```powershell
.\setup.ps1 -CondaEnv qwen35
.\setup.ps1 -Python C:\Python312\python.exe
```

```bash
./setup.sh --conda-env qwen35
./setup.sh --python /usr/bin/python3.12
```

Pass `-Minimal` / `--minimal` for the core packages only; the deep-learning
arms then report `UNVERIFIED`, which is the designed behaviour rather than a
failure.

### Missing, broken, and the difference between them

A package that is **absent** makes its arm `UNVERIFIED` — an honest skip. A
package that is **installed but fails to import** makes its arm `BUG`, keeps
the error text in the report, and drops the whole run to `PENDING REVISION`.
The two are not the same condition and the testbed never conflates them:

```bash
python -m mlkit.verify --env       # inventory, with BROKEN called out
python -m mlkit.verify --missing   # pip names of absent packages only
```

## Layout

```
testbed/
  mlkit/
    data.py        the shared input field -- the only way data enters
    metrics.py     the single measurement tools -- the only scorer
    splits.py      splitters plus the assertions that catch leaks
    report.py      run reports: the audit pack
    config.py      the control panel (--set, --sweep, --compare)
    verify.py      the Auditor's entry point
  experiments/
    exp01_baseline.py   the single baseline experiment
  panel/
    reports.py     parses runs/ into one JSON-safe payload
    serve.py       the read-only dashboard, `python -m panel.serve`
    static/        vanilla HTML/CSS/JS, no build step
  exercises/
    stage1_linalg.py    Stage 1: you implement these
  tests/
    test_metrics.py     the measurement contract
    test_splits.py      the split contract
    test_leakage.py     the leakage contract, demonstrated on noise
    test_stage1_gates.py  Stage 1 acceptance gates (skip until implemented)
    test_panel_reports.py the panel's parsing contract
  configs/exp01.toml    controls for the baseline
  runs/                 one audit pack per run (created on first run)
```

## Daily commands

```bash
# run the baseline and write an audit pack
python -m experiments.exp01_baseline --config configs/exp01.toml

# change a parameter without touching the source
python -m experiments.exp01_baseline --config configs/exp01.toml \
    --set arms.gradient_descent.lr=0.3

# scan a parameter range; divergence above the stability limit is the
# point of this one [MML p.381]
python -m experiments.exp01_baseline --config configs/exp01.toml \
    --sweep arms.gradient_descent.lr=0.01,0.2,0.5,0.9,1.1

# read an arm's implementation and its page anchors
python -m experiments.exp01_baseline --config configs/exp01.toml \
    --show-source gradient_descent

# read the runs in a browser (read-only dashboard)
python -m panel.serve

# what can this machine verify?
python -m mlkit.verify --env

# re-check a finished run as the Auditor, from the run directory alone
python -m mlkit.verify runs/<run-id>

# contracts, and the Stage 1 gates
pytest tests -q
pytest tests/test_stage1_gates.py -v

# the code contract [GPSG]
ruff check . && ruff format --check . && mypy mlkit
```

## The two rules

1. **Nothing computes its own metric.** Every number comes from
   `mlkit.metrics`, because three of the source books define overlapping
   metrics that disagree at the edges (ledger GAP-09).
2. **Nothing loads its own data.** Every array comes from
   `mlkit.data.load_dataset`, so two methodologies are always compared on
   byte-identical inputs.

Break either one and the comparison tables stop meaning anything, quietly.

## What a run leaves behind

```
runs/20261017T005554-exp01_baseline/
  config.json        the resolved config, including any --set overrides
  report.json        machine-readable: arms, gates, gaps, environment,
                     and created_at, the time the run was written
  report.md          human-readable summary with status tags
  env.txt            interpreter, platform, package versions
  dataset_card.json  name, shape, seed, checksum, provenance
```

That directory is the audit pack [SKL ch.14]. If a reviewer cannot tell what
you ran from those files alone, the run is not finished.

## Panel

A read-only dashboard over `runs/`. Standard library only: a small static
server plus vanilla HTML, CSS and JS. No build step, no database, no new
runtime dependency.

```bash
# open the dashboard against the existing runs/
python -m panel.serve

# another port, no browser window, no request log
python -m panel.serve --port 9000 --no-browser --quiet

# the same answers in a terminal; exits 1 if any run is not VERIFIED
python -m panel.serve --check
```

| View | What it answers |
|---|---|
| Status board | what was run, when, on which data, and is it trusted |
| Run detail | arms with 95% CIs and wall-clock seconds, gates with their page anchors, split boundaries, arm notes, gaps, environment, resolved config |
| Compare | two or more runs of one experiment: config diff beside the primary metric, so a parameter change is traceable to a metric change |
| Sweeps | runs differing in exactly one config key, that key plotted against the primary metric |
| Tags | what the four tags mean, and the two markers that are not tags |

Five rules hold it together.

1. **No number without its tag.** Every metric reaches the page through
   one function that cannot render a value without the status of the arm
   that produced it. There is deliberately no helper that returns a bare
   formatted number, because the moment one exists some table uses it.
2. **Nothing is invented.** A field the report does not carry is reported
   missing, never defaulted. A `report.json` that will not parse is
   marked `UNREADABLE`, which is a parse outcome and emphatically not a
   fifth status tag: the panel does not get to decide a run's status. An
   arm the config declared but the report never produced is shown as a
   dropped arm rather than omitted.
3. **Unverified stays in view.** A banner above every view names every
   run that is not `VERIFIED`, including unreadable ones. There is no
   filter that takes it away.
4. **Divergence is a result, not an axis.** A metric that is non-finite,
   or above `1e6` in magnitude, is drawn in a separate band above the
   plot with no position on the axis at all, and the series line breaks
   there. Otherwise one `rmse = 4.9e+65` flattens every other point into
   a single line. Divergence above the stability limit is a required
   result of the `lr` sweep, not a failure [MML p.381], [MML p.389].
5. **It is a viewer.** The server answers `GET` and `HEAD` and nothing
   else, and binds to loopback. Nothing in the panel edits a run, a
   threshold or a status -- editing a threshold after seeing a result is
   the bug this program exists to prevent (VERIFICATION.md §2.6).

The browser is never handed a raw `report.json`. A diverged run writes
`NaN`, which Python's `json` module emits happily and the browser's
`JSON.parse` rejects outright. `panel/reports.py` normalises every pack
into one JSON-safe payload served at `/api/runs.json`, rebuilt from disk
on each request, so a run that finished a second ago appears on the next
refresh.

### Feeding the sweep view

`--sweep` prints a parameter/metric curve but writes no audit packs, so
the sweep view reads runs made with repeated `--set`:

```bash
for lr in 0.01 0.05 0.2 0.5 0.9 1.1; do
  python -m experiments.exp01_baseline --config configs/exp01.toml \
      --set arms.gradient_descent.lr=$lr
done
```

The panel groups those runs by the one key that varies. Runs differing
in two keys form no sweep, because such a pair explains nothing.

### One field added to the audit pack

`report.json` gained `created_at` (ISO 8601), written by
`mlkit.report.write_run` from the same moment it stamps the run
directory with. The change is additive: reports written before it exists
parse exactly as before, and the panel falls back to the directory name,
marking those timestamps with `*` on the status board. The reason is the
audit-pack rule itself -- a reviewer holding `report.json` alone should
not have to reconstruct when the run happened from a directory name,
which is a naming convention rather than data.
