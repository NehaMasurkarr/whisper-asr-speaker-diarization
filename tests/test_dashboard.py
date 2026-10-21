from pathlib import Path
import pytest


def test_dashboard_loads():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).parents[1] / "src/meeting_asr/dashboard.py"))
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Meeting Speech Lab"
