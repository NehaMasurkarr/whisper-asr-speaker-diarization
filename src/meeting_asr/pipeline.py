from pathlib import Path
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
import json
from .audio import load_audio
from .alignment import align
from .config import Config
from .inference import transcribe, diarize
from .types import Transcript


def run(
    path: Path,
    config: Config,
    start: float = 0,
    duration: float | None = None,
    asr_only: bool = False,
) -> Transcript:
    audio = load_audio(path, start, duration)
    # Fail before expensive ASR if diarization authentication is absent.
    if not asr_only and not config.hf_token:
        from .inference import ModelAccessError

        raise ModelAccessError("Set HF_TOKEN and accept Community-1 terms, or use --asr-only.")
    words, asr_device = transcribe(audio, config)
    turns, diar_device = ([], None) if asr_only else diarize(audio, config)
    packages = {}
    for package in ("openai-whisper", "pyannote.audio", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    return Transcript(
        align(words, turns),
        turns,
        {
            "schema_version": 1,
            "audio_name": path.name,
            "offset_seconds": start,
            "duration_seconds": len(audio) / 16000,
            "timestamp_origin": "crop",
            "whisper_model": config.whisper_model,
            "diarization_model": config.diarization_model,
            "asr_only": asr_only,
            "asr_device": asr_device,
            "diarization_device": diar_device,
            "beam_size": config.beam_size,
            "cpu_threads": config.cpu_threads,
            "diarization_batch_size": config.diarization_batch_size,
            "num_speakers": config.num_speakers,
            "initial_prompt": config.initial_prompt,
            "packages": packages,
        },
    )


def readable(transcript: Transcript) -> str:
    lines = []
    previous = None
    for word in transcript.words:
        label = "/".join(word.speakers) or "UNKNOWN"
        key = (label, word.overlap)
        if key != previous:
            lines.append(f"\n[{word.start:.2f}s] {label}{' [overlap]' if word.overlap else ''}:")
        lines[-1] += " " + word.text.strip()
        previous = key
    return "\n".join(lines).lstrip() + "\n"


def save(transcript: Transcript, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.json").write_text(
        json.dumps(asdict(transcript), indent=2), encoding="utf-8"
    )
    (directory / "transcript.txt").write_text(readable(transcript), encoding="utf-8")
