from types import SimpleNamespace
import sys
import pytest
from meeting_asr.config import Config
from meeting_asr.inference import transcribe, diarize, ModelAccessError


def test_whisper_mps_fallback(monkeypatch):
    devices = []

    def load(name, device, **kwargs):
        devices.append(device)
        if device == "mps":
            raise NotImplementedError("unsupported operation")
        return SimpleNamespace(
            transcribe=lambda *a, **kw: {
                "segments": [{"words": [{"start": 0, "end": 1, "word": "hello"}]}]
            }
        )

    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=load))
    words, actual = transcribe(None, Config(asr_device="mps"))
    assert actual == "cpu" and devices == ["mps", "cpu"]
    assert words[0].text == "hello"


def test_whisper_cpu_failure_propagates(monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("out of memory")

    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=fail))
    with pytest.raises(RuntimeError, match="out of memory"):
        transcribe(None, Config(asr_device="cpu"))


def test_missing_auth():
    with pytest.raises(ModelAccessError, match="HF_TOKEN"):
        diarize(None, Config(hf_token=None))


def test_config_token_redacted():
    assert "secret" not in repr(Config(hf_token="secret"))
    with pytest.raises(ValueError):
        Config(num_speakers=0)
