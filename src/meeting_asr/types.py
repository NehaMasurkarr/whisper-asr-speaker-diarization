from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str

    def __post_init__(self):
        if not all(map(math.isfinite, (self.start, self.end))) or not 0 <= self.start < self.end:
            raise ValueError("Turn requires finite timestamps and 0 <= start < end")


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    speakers: tuple[str, ...] = ()
    overlap: bool = False

    def __post_init__(self):
        if not all(map(math.isfinite, (self.start, self.end))) or not 0 <= self.start <= self.end:
            raise ValueError("Word requires finite timestamps and 0 <= start <= end")


@dataclass
class Transcript:
    words: list[Word]
    turns: list[Turn]
    metadata: dict = field(default_factory=dict)
