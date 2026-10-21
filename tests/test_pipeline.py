from dataclasses import asdict
import json
import numpy as np
import pytest
import soundfile as sf
from meeting_asr.types import Turn, Word, Transcript
from meeting_asr.alignment import align
from meeting_asr.audio import load_audio
from meeting_asr.config import Config
from meeting_asr.evaluation import load_ami, normalize
from meeting_asr.pipeline import run, save


def test_overlap_and_change():
    turns = [Turn(0, 1, "A"), Turn(1, 2, "B"), Turn(1.5, 2, "C")]
    output = align([Word(0.9, 1.1, "change"), Word(1.6, 1.9, "overlap"), Word(3, 4, "gap")], turns)
    assert set(output[0].speakers) == {"A", "B"}
    assert not output[0].overlap
    assert output[1].overlap and set(output[1].speakers) == {"B", "C"}
    assert output[2].speakers == ()


def test_boundaries():
    assert align([Word(1, 1, "x")], [Turn(0, 1, "A"), Turn(1, 2, "B")])[0].speakers == ("B",)
    with pytest.raises(ValueError):
        Turn(2, 1, "A")
    with pytest.raises(ValueError):
        Word(float("nan"), 1, "x")


def test_audio_and_pipeline(tmp_path, monkeypatch):
    path = tmp_path / "test.wav"
    sf.write(path, np.zeros((8000, 2)), 8000)
    assert load_audio(path, duration=0.5).shape == (8000,)
    monkeypatch.setattr(
        "meeting_asr.pipeline.transcribe", lambda *a: ([Word(0, 0.4, "hello")], "cpu")
    )
    monkeypatch.setattr("meeting_asr.pipeline.diarize", lambda *a: ([Turn(0, 0.5, "A")], "cpu"))
    transcript = run(path, Config(hf_token="test-secret"), duration=0.5)
    save(transcript, tmp_path / "out")
    data = json.loads((tmp_path / "out/transcript.json").read_text())
    assert data["words"][0]["speakers"] == ["A"]
    assert "test-secret" not in json.dumps(asdict(transcript))
    assert "hello" in (tmp_path / "out/transcript.txt").read_text()


def test_missing_token(tmp_path):
    path = tmp_path / "test.wav"
    sf.write(path, np.zeros(1600), 16000)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        run(path, Config(hf_token=None))


def test_ami(tmp_path):
    (tmp_path / "ES2002a.A.words.xml").write_text(
        '<root><w starttime="0" endtime="1">Hello</w><w punc="true" starttime="1" endtime="2">.</w></root>'
    )
    words, turns = load_ami(tmp_path, "ES2002a", activity="words")
    assert len(words) == len(turns) == 1
    assert normalize("Hello, WORLD!") == "hello world"
    with pytest.raises(FileNotFoundError):
        load_ami(tmp_path, "missing")


def test_metrics_permutation_overlap_and_crop():
    pytest.importorskip("pyannote.metrics")
    from meeting_asr.evaluation import evaluate

    transcript = Transcript(
        [Word(0, 1, "hello"), Word(1, 2, "world")],
        [Turn(0, 2, "X"), Turn(1, 2, "Y")],
        {"offset_seconds": 10, "duration_seconds": 2},
    )
    words = [Word(10, 11, "hello"), Word(11, 12, "world")]
    refs = [Turn(10, 12, "A"), Turn(11, 12, "B")]
    metrics = evaluate(transcript, words, refs)
    assert metrics["wer"] == 0 and metrics["der"] == 0
    transcript.turns = [Turn(0, 2, "X")]
    assert evaluate(transcript, words, refs)["der"] == pytest.approx(1 / 3)
    transcript.words = [Word(0, 1, "wrong")]
    assert evaluate(transcript, words, refs)["wer"] == 1


def test_ami_segments_required_and_loaded(tmp_path):
    (tmp_path / "M.A.words.xml").write_text('<root><w starttime="1" endtime="2">Hi</w></root>')
    with pytest.raises(FileNotFoundError, match="segment"):
        load_ami(tmp_path, "M")
    (tmp_path / "M.A.segments.xml").write_text(
        '<root><segment transcriber_start="0.8" transcriber_end="2.2"/></root>'
    )
    _, turns = load_ami(tmp_path, "M")
    assert turns == [Turn(0.8, 2.2, "A")]


def test_audio_invalid_and_export_empty(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(100), 16000)
    with pytest.raises(ValueError):
        load_audio(path, start=-1)
    with pytest.raises(ValueError):
        load_audio(path, duration=0)
    with pytest.raises(ValueError):
        load_audio(path, duration=float("inf"))
    save(Transcript([], []), tmp_path / "out")
    assert json.loads((tmp_path / "out/transcript.json").read_text())["words"] == []
