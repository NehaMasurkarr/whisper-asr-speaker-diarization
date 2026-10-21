import json
import sys
from meeting_asr.cli import main
from meeting_asr.types import Transcript, Word


def test_cli_exports_asr_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "meeting_asr.cli.run",
        lambda *a, **kw: Transcript(
            [Word(0, 1, "hello")], [], {"offset_seconds": 0, "duration_seconds": 1}
        ),
    )
    monkeypatch.setattr(
        sys, "argv", ["meeting-asr", "input.wav", "--asr-only", "--output", str(tmp_path)]
    )
    main()
    assert json.loads((tmp_path / "transcript.json").read_text())["words"][0]["text"] == "hello"


def test_logging_escapes_message():
    import logging
    from meeting_asr.logging import JsonFormatter

    record = logging.LogRecord("test", logging.INFO, "", 0, 'a "quoted"\nmessage', (), None)
    output = JsonFormatter().format(record)
    assert json.loads(output)["event"] == 'a "quoted"\nmessage'
    assert len(output.splitlines()) == 1


def test_invalid_environment_is_actionable(monkeypatch, capsys):
    import pytest

    monkeypatch.setenv("TORCH_NUM_THREADS", "0")
    monkeypatch.setattr(sys, "argv", ["meeting-asr", "input.wav"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "TORCH_NUM_THREADS must be positive" in capsys.readouterr().err
