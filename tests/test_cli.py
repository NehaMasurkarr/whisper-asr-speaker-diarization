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
