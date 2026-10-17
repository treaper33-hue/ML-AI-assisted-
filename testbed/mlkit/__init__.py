"""Shared toolkit for the ML mastery testbed.

This package exists so that every experiment in the program shares exactly
three things, as required by the Prompt Guidelines:

  * one input field    -- `mlkit.data.load_dataset`
  * one set of metrics -- `mlkit.metrics`
  * one set of splits  -- `mlkit.splits`

Experiments must never compute their own metrics or load their own files.
That rule is what makes cross-methodology comparison meaningful.
"""

__version__ = '1.0.0'

STATUS_VERIFIED = 'VERIFIED'
STATUS_PENDING = 'PENDING REVISION'
STATUS_UNVERIFIED = 'UNVERIFIED'
STATUS_BUG = 'BUG'
