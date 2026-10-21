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


def test_pyannote_waveform_adapter_and_fallback(monkeypatch):
    import numpy as np
    import torch
    from meeting_asr.types import Turn

    moved = []
    calls = []

    class FakePipeline:
        def to(self, target):
            moved.append(str(target))

        def __call__(self, waveform, **kwargs):
            calls.append(kwargs)
            assert waveform["waveform"].shape == (1, 16000)
            assert waveform["sample_rate"] == 16000
            if moved[-1] == "mps":
                raise RuntimeError("unsupported")
            annotation = SimpleNamespace(
                itertracks=lambda **kw: iter(
                    [(SimpleNamespace(start=0, end=1), None, "SPEAKER_00")]
                )
            )
            return SimpleNamespace(speaker_diarization=annotation)

    fake = FakePipeline()
    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=lambda *a, **kw: fake)),
    )
    turns, actual = diarize(
        np.zeros(16000, dtype="float32"),
        Config(hf_token="private", diarization_device="mps", num_speakers=4),
    )
    assert actual == "cpu" and moved == ["mps", "cpu"]
    assert turns == [Turn(0, 1, "SPEAKER_00")]
    assert calls == [{"num_speakers": 4}] * 2
    assert torch.is_tensor(torch.from_numpy(np.zeros(1)))


def test_pyannote_access_error_redacts_upstream(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("credential=supersecret")

    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=fail)),
    )
    with pytest.raises(ModelAccessError) as error:
        diarize(None, Config(hf_token="supersecret"))
    assert "supersecret" not in str(error.value)
