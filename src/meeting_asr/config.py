from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class Config:
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base.en"))
    asr_device: str = field(default_factory=lambda: os.getenv("ASR_DEVICE", "auto"))
    diarization_device: str = field(default_factory=lambda: os.getenv("DIARIZATION_DEVICE", "auto"))
    diarization_model: str = field(
        default_factory=lambda: os.getenv(
            "DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
        )
    )
    diarization_revision: str | None = field(
        default_factory=lambda: os.getenv("DIARIZATION_REVISION")
    )
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"), repr=False)
    diarization_batch_size: int = field(
        default_factory=lambda: int(os.getenv("DIARIZATION_BATCH_SIZE", "4"))
    )
    cpu_threads: int = field(default_factory=lambda: int(os.getenv("TORCH_NUM_THREADS", "4")))
    num_speakers: int | None = None
    beam_size: int = 5
    initial_prompt: str | None = None

    def __post_init__(self):
        if self.asr_device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("Unsupported ASR device")
        if self.diarization_device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("Unsupported diarization device")
        if self.diarization_batch_size < 1:
            raise ValueError("DIARIZATION_BATCH_SIZE must be positive")
        if self.cpu_threads < 1:
            raise ValueError("TORCH_NUM_THREADS must be positive")
        if self.beam_size < 1 or (self.num_speakers is not None and self.num_speakers < 1):
            raise ValueError("Beam size and speaker count must be positive")
