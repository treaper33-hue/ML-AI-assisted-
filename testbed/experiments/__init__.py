"""Experiments.

One module per experiment. Each one reads a config, loads data through
`mlkit.data`, scores with `mlkit.metrics`, declares its gates up front, and
writes an audit pack to `runs/`.

Current experiments:
    exp01_baseline  the single baseline experiment (Prompt Guidelines, 5)

Planned, one per stage gate:
    exp02_tabular   leak-proof classical benchmark   (Stage 2)
    exp03_deep      hand-written training loop       (Stage 3)
    exp04_forecast  rolling-origin forecasting       (Stage 4)
"""
