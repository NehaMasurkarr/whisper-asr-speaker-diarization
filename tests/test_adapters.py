from types import SimpleNamespace
import sys
import pytest
from meeting_asr.config import Config
from meeting_asr.inference import transcribe, diarize, ModelAccessError


@pytest.fixture(autouse=True)
def checkpoint_metadata_stubs(monkeypatch):
    monkeypatch.setattr("meeting_asr.inference._checkpoint_types", lambda: [])


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
    with pytest.raises(ValueError):
        Config(diarization_batch_size=0)


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
            assert self.embedding_batch_size == 4
            assert self.segmentation_batch_size == 4
            assert waveform["waveform"].shape == (1, 16000)
            assert waveform["sample_rate"] == 16000
            if moved[-1] == "mps":
                raise RuntimeError("unsupported")
            annotation = SimpleNamespace(
                itertracks=lambda **kw: iter(
                    [
                        (SimpleNamespace(start=-0.01, end=1.1), None, "SPEAKER_00"),
                        (SimpleNamespace(start=1.1, end=1.2), None, "PADDED"),
                    ]
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


class CheckpointMetadata:
    def __init__(self, value):
        self.value = value


def test_scoped_checkpoint_allowlist(tmp_path, monkeypatch):
    import numpy as np
    import torch

    checkpoint = tmp_path / "metadata.pt"
    torch.save({"metadata": CheckpointMetadata(7)}, checkpoint)
    before = list(torch.serialization.get_safe_globals())

    def load(*args, **kwargs):
        restored = torch.load(checkpoint, weights_only=True)
        assert restored["metadata"].value == 7
        return SimpleNamespace(
            to=lambda *a: None,
            __call__=lambda *a, **kw: None,
        )

    class FakePipeline:
        def to(self, target):
            pass

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(speaker_diarization=SimpleNamespace(itertracks=lambda **kw: []))

    def factory(*args, **kwargs):
        load()
        return FakePipeline()

    monkeypatch.setattr("meeting_asr.inference._checkpoint_types", lambda: [CheckpointMetadata])
    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=factory)),
    )
    diarize(np.zeros(16000, dtype="float32"), Config(hf_token="test", diarization_device="cpu"))
    assert torch.serialization.get_safe_globals() == before
