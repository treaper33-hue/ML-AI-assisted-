"""The visual control panel: a read-only viewer over `runs/`.

The command line already covers the controls listed in
ML_LEARNING_PROGRAM.md §3. This package adds the other half of
Guideline 4 -- seeing a parameter change next to the metric change it
caused -- without becoming a second source of truth.

Three rules hold the package together:

  * **It never writes.** No run, threshold or status is editable from
    here. Editing a threshold after seeing a result is the bug this
    program is built to prevent [VERIFICATION.md §2.6].
  * **No number without its tag.** Every metric is normalised into a
    cell that travels with the owning arm's status, so a renderer
    cannot show a value without showing how far it can be trusted.
  * **Nothing is invented.** A field the report does not carry comes
    back as None with an entry in `errors`, never as a default that
    reads like a measurement.

Modules:
    reports: parsing and normalising the audit packs.
    serve:   the static-file and JSON server, `python -m panel.serve`.
"""

__version__ = '1.0.0'
