from pathlib import Path
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: Path, start: float = 0, duration: float | None = None) -> np.ndarray:
    if start < 0 or (duration is not None and duration <= 0):
        raise ValueError("Invalid audio crop")
    with sf.SoundFile(path) as source:
        rate = source.samplerate
        source.seek(round(start * rate))
        audio = source.read(
            -1 if duration is None else round(duration * rate), dtype="float32", always_2d=True
        )
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains non-finite samples")
    mono = audio.mean(axis=1)
    divisor = gcd(rate, 16000)
    return resample_poly(mono, 16000 // divisor, rate // divisor).astype("float32")
