"""Smoke tests for the Streamlit app.

These execute ``app.py`` through Streamlit's own test harness, so a runtime
error in the UI layer fails the suite rather than surfacing to a user. The
whole module is skipped when Streamlit is not installed, keeping the core
analytical suite runnable in a minimal environment.

Run from the repository root::

    pytest tests/test_app.py -q
"""

from __future__ import annotations

import os

import pytest

streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "app.py")
TIMEOUT = 300


def _run(**slider_or_radio):
    """Load the app, optionally change a widget, and assert it never raises."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    if slider_or_radio:
        for idx, value in slider_or_radio.get("sliders", {}).items():
            at.slider[idx].set_value(value)
        if "radio" in slider_or_radio:
            at.radio[0].set_value(slider_or_radio["radio"])
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception():
    at = _run()
    assert at.title[0].value.endswith("PlasmidQC-Studio")
    assert len(at.tabs) == 4
    assert len(at.dataframe) == 3


def test_app_reports_correct_default_metrics():
    m = _metrics(_run())
    assert m["Length"] == "3,161 bp"
    assert m["GC content"] == "50.11%"
    assert m["Features"] == "8"
    assert m["Unique cutters"] == "6"
    assert m["Primer pairs"] == "5"


def test_app_min_orf_slider_changes_feature_count():
    """Lowering the ORF threshold must expose more CDS calls, raising it fewer."""
    permissive = int(_metrics(_run(sliders={0: 30}))["Features"])
    default = int(_metrics(_run())["Features"])
    strict = int(_metrics(_run(sliders={0: 400}))["Features"])
    assert permissive > default > strict


def test_app_primer_flank_and_gc_window_sliders_are_safe():
    assert _metrics(_run(sliders={1: 150}))["Primer pairs"] == "5"
    assert _metrics(_run(sliders={2: 400, 3: 100}))["Length"] == "3,161 bp"


def test_app_second_bundled_plasmid_loads():
    m = _metrics(_run(radio="Bundled: pBioBrick_mCherry_AmpR"))
    assert m["Length"] == "3,163 bp"
    assert m["Unique cutters"] == "5"


def test_app_upload_mode_prompts_instead_of_crashing():
    at = _run(radio="Upload FASTA")
    assert at.info, "expected an informational prompt when no file is uploaded"
    assert not at.exception
