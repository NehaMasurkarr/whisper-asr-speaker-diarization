from pathlib import Path
import pytest


def test_dashboard_loads():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).parents[1] / "src/meeting_asr/dashboard.py"))
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Meeting Speech Lab"


def test_dashboard_renders_results():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from meeting_asr.types import Transcript, Word, Turn

    app = AppTest.from_file(str(Path(__file__).parents[1] / "src/meeting_asr/dashboard.py"))
    app.session_state["transcript"] = Transcript([Word(0, 1, "hello", ("A",))], [Turn(0, 1, "A")])
    app.session_state["metrics"] = {"wer": 0.2, "der": None}
    app.run(timeout=30)
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["20.00%", "Unavailable"]
