"""AMI word XML parsing and explicitly scoped, overlap-inclusive evaluation."""

from pathlib import Path
import re
from defusedxml import ElementTree
from jiwer import process_words
from .types import Word, Turn, Transcript


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+(?:'\w+)?", text.lower()))


def load_ami(
    directory: Path, meeting: str, activity: str = "segments"
) -> tuple[list[Word], list[Turn]]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", meeting):
        raise ValueError("Invalid meeting identifier")
    if activity not in {"segments", "words"}:
        raise ValueError("Unknown reference activity")
    files = sorted(directory.rglob(f"{meeting}.*.words.xml"))
    if not files:
        raise FileNotFoundError(f"No {meeting}.*.words.xml annotations in {directory}")
    words, turns = [], []
    for path in files:
        speaker = path.name[len(meeting) + 1 : -len(".words.xml")]
        for node in ElementTree.parse(path).getroot().iter():
            if node.tag.split("}")[-1] != "w" or not {"starttime", "endtime"} <= node.attrib.keys():
                continue
            if node.attrib.get("punc", "false").lower() == "true":
                continue
            text = "".join(node.itertext()).strip()
            start, end = float(node.attrib["starttime"]), float(node.attrib["endtime"])
            if text and end > start:
                words.append(Word(start, end, text, (speaker,)))
                turns.append(Turn(start, end, speaker))
    if not words:
        raise ValueError("Annotations contain no timed lexical words")
    if activity == "segments":
        turns = []
        for word_file in files:
            speaker = word_file.name[len(meeting) + 1 : -len(".words.xml")]
            matches = list(directory.rglob(f"{meeting}.{speaker}.segments.xml"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one segment annotation for {meeting}.{speaker}")
            for node in ElementTree.parse(matches[0]).getroot().iter():
                if node.tag.split("}")[-1] == "segment":
                    turns.append(
                        Turn(
                            float(node.attrib["transcriber_start"]),
                            float(node.attrib["transcriber_end"]),
                            speaker,
                        )
                    )
        if not turns:
            raise ValueError("No reference speech segments")
    return sorted(words, key=lambda w: (w.start, w.end, w.speakers)), turns


def evaluate(
    transcript: Transcript,
    reference_words: list[Word],
    reference_turns: list[Turn],
    collar: float = 0.0,
    skip_overlap: bool = False,
    reference_type: str = "AMI transcriber speech segments",
) -> dict:
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    start = float(transcript.metadata["offset_seconds"])
    end = start + float(transcript.metadata["duration_seconds"])
    if collar < 0 or end <= start:
        raise ValueError("Invalid scoring region or collar")
    refs = [w for w in reference_words if start <= (w.start + w.end) / 2 < end]
    if not refs:
        raise ValueError("No reference words in scoring region")
    ref_text = normalize(" ".join(w.text for w in refs))
    hyp_text = normalize(" ".join(w.text for w in transcript.words))
    wer = process_words(ref_text, hyp_text)

    def annotation(turns, offset=0):
        result = Annotation()
        for i, turn in enumerate(turns):
            a, b = max(start, turn.start + offset), min(end, turn.end + offset)
            if b > a:
                result[Segment(a, b), i] = turn.speaker
        return result

    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    details = metric(
        annotation(reference_turns),
        annotation(transcript.turns, start),
        uem=Timeline([Segment(start, end)]),
        detailed=True,
    )
    return {
        "wer": wer.wer,
        "substitutions": wer.substitutions,
        "deletions": wer.deletions,
        "insertions": wer.insertions,
        "reference_words": len(ref_text.split()),
        "der": float(details["diarization error rate"]),
        "der_components_seconds": {
            str(k): float(v) for k, v in details.items() if k != "diarization error rate"
        },
        "collar_seconds": collar,
        "skip_overlap": skip_overlap,
        "uem": [start, end],
        "reference_type": reference_type,
        "wer_order": "reference words sorted by start; hypothesis in decoding order",
    }
