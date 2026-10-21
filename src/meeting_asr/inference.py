"""Lazy model adapters: importing core modules never downloads weights."""

import logging
import os
from .config import Config
from .types import Word, Turn

log = logging.getLogger(__name__)


class ModelAccessError(RuntimeError):
    pass


def device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def transcribe(audio, config: Config) -> tuple[list[Word], str]:
    import whisper
    import torch

    torch.set_num_threads(config.cpu_threads)

    selected = device(config.asr_device)

    def run(target):
        model = whisper.load_model(
            config.whisper_model,
            device=target,
            download_root=os.getenv("WHISPER_CACHE", ".cache/whisper"),
        )
        output = model.transcribe(
            audio,
            word_timestamps=True,
            fp16=target == "cuda",
            beam_size=config.beam_size,
            initial_prompt=config.initial_prompt,
            temperature=0,
            verbose=False,
        )
        return [
            Word(float(w["start"]), float(w["end"]), w["word"])
            for segment in output["segments"]
            for w in segment.get("words", [])
        ]

    try:
        return run(selected), selected
    except (RuntimeError, NotImplementedError):
        if selected != "mps":
            raise
        log.warning("asr_mps_failed_retrying_cpu")
    # Exit the exception scope first so its traceback releases failed MPS tensors.
    import gc
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return run("cpu"), "cpu"


def _checkpoint_types() -> list[type]:
    # Only known pyannote metadata classes are permitted during checkpoint loading.
    from pyannote.audio.core.task import Specifications, Problem, Resolution

    return [Specifications, Problem, Resolution]


def diarize(audio, config: Config) -> tuple[list[Turn], str]:
    if not config.hf_token:
        raise ModelAccessError(
            "Set HF_TOKEN and accept the diarization model's Hugging Face terms."
        )
    import torch

    torch.set_num_threads(config.cpu_threads)
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    from pyannote.audio import Pipeline

    try:
        with torch.serialization.safe_globals(_checkpoint_types()):
            pipeline = Pipeline.from_pretrained(config.diarization_model, token=config.hf_token)
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        detail = f" ({type(error).__name__}{f', HTTP {status}' if status else ''})"
        raise ModelAccessError(
            "Unable to load diarization model"
            + detail
            + ". Check model access, dependencies, and network."
        ) from None
    if pipeline is None:
        raise ModelAccessError(
            "Diarization model access denied. Check model terms and token permissions."
        )
    pipeline.embedding_batch_size = config.diarization_batch_size
    pipeline.segmentation_batch_size = config.diarization_batch_size
    selected = device(config.diarization_device)
    kwargs = {"num_speakers": config.num_speakers} if config.num_speakers else {}
    waveform = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}
    try:
        pipeline.to(torch.device(selected))
        output = pipeline(waveform, **kwargs)
    except (RuntimeError, NotImplementedError):
        if selected != "mps":
            raise
        log.warning("diarization_mps_failed_retrying_cpu")
        selected = "cpu"
        pipeline.to(torch.device(selected))
        output = pipeline(waveform, **kwargs)
    annotation = output.speaker_diarization
    return [
        Turn(float(s.start), float(s.end), str(label))
        for s, _, label in annotation.itertracks(yield_label=True)
    ], selected
