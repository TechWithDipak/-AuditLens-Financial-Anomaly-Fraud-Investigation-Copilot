"""Headless end-to-end testing of AuditLens Streamlit application using AppTest."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_app_loads_and_runs_without_exceptions():
    """Verify app initializes, runs headless, and renders without exceptions."""
    app_path = Path("app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    assert not at.exception, f"App threw exception on initial load: {at.exception}"
    assert at.title[0].value == "AuditLens"

    # Verify 4 tabs
    tab_labels = [tab.label for tab in at.tabs]
    assert tab_labels == ["Overview", "Transactions", "Algorithms", "How it works"]


def test_tab_overview_metrics_rendered():
    """Verify key metrics and tables are present in Overview tab."""
    app_path = Path("app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    assert not at.exception
    # Check that Overview contains markdown and confusion matrix table
    assert len(at.table) >= 1
    # Check that headers are present
    headings = [h.value for h in at.subheader]
    assert "Evaluation Summary" in headings
    assert "Risk Score Distribution" in headings
    assert "Confusion Matrix" in headings


def test_tab_transactions_selection_and_report():
    """Verify transaction selection, why-flagged text, and download button."""
    app_path = Path("app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    assert not at.exception

    # Selectboxes for verdict, category, and transaction
    assert len(at.selectbox) >= 3
    # Download button exists
    assert len(at.download_button) >= 1
    dl_btn = at.download_button[0]
    assert "Download audit report" in dl_btn.label


def test_tab_transactions_direct_investigation_option():
    """Verify that table has direct Investigation column and updates active selection."""
    app_path = Path("app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    assert not at.exception
    # Verify dataframe element exists and has Investigation column
    assert len(at.dataframe) >= 1
    df_elem = at.dataframe[0]
    assert "Investigation" in df_elem.value.columns
    assert df_elem.value.columns.get_loc("Investigation") == df_elem.value.columns.get_loc("Risk Score") + 1

    # Trigger investigation click via session state
    at.session_state["investigate_action"] = {"row": 1, "label": "Investigate"}
    at.run()
    assert not at.exception
    assert at.session_state.get("selected_txn_id") is not None


def test_tab_algorithms_interactive_execution():
    """Verify pressing every algorithm run button executes properly without exceptions."""
    app_path = Path("app.py").resolve()
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    assert not at.exception

    # Buttons:
    # 0: Run PCA 2D projection on sample
    # 1: Run EM optimization on PCA sample
    # 2: Run Forward and Viterbi Verification against hmmlearn
    for i in range(len(at.button)):
        btn = at.button[i]
        label = btn.label
        btn.click()
        at.run()
        assert not at.exception, f"Exception occurred after clicking button {i} ('{label}'): {at.exception}"


def test_app_runs_in_environment_without_hmmlearn():
    """Verify app runs without throwing exceptions when hmmlearn is missing from the environment."""
    import subprocess
    import sys

    code = """
import sys
class DisallowHmmlearn:
    def find_spec(self, fullname, path, target=None):
        if 'hmmlearn' in fullname:
            raise ModuleNotFoundError(f'No module named {fullname}')
        return None
sys.meta_path.insert(0, DisallowHmmlearn())
for m in list(sys.modules.keys()):
    if 'hmmlearn' in m:
        del sys.modules[m]

from streamlit.testing.v1 import AppTest
at = AppTest.from_file('app.py', default_timeout=30)
at.run()
assert not at.exception, f'App crashed with: {at.exception}'
print('SUCCESS_NO_HMMLEARN')
"""
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"App execution failed without hmmlearn:\n{res.stderr}\n{res.stdout}"
    assert "SUCCESS_NO_HMMLEARN" in res.stdout

