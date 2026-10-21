"""Attribute words by intersecting temporal support; preserve simultaneous candidates."""

from dataclasses import replace
from .types import Turn, Word


def align(words: list[Word], turns: list[Turn]) -> list[Word]:
    result = []
    for word in words:
        candidates: dict[str, float] = {}
        active = []
        for turn in turns:
            duration = min(word.end, turn.end) - max(word.start, turn.start)
            if duration > 0 or (word.start == word.end and turn.start <= word.start < turn.end):
                candidates[turn.speaker] = candidates.get(turn.speaker, 0) + max(duration, 0)
                active.append(turn)
        simultaneous = any(
            a.speaker != b.speaker
            and max(a.start, b.start, word.start) < min(a.end, b.end, word.end)
            for i, a in enumerate(active)
            for b in active[i + 1 :]
        )
        ranked = tuple(sorted(candidates, key=lambda s: (-candidates[s], s)))
        result.append(replace(word, speakers=ranked, overlap=simultaneous))
    return result
