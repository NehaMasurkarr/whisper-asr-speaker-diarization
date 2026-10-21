import argparse
from dataclasses import replace
import json
from pathlib import Path
from dotenv import load_dotenv
from .config import Config
from .logging import configure_logging
from .pipeline import run, save
from .evaluation import load_ami, evaluate


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Local Whisper + pyannote meeting pipeline")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--model", default=None)
    parser.add_argument("--speakers", type=int)
    parser.add_argument("--prompt")
    parser.add_argument("--asr-only", action="store_true")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--meeting", default="ES2002a")
    parser.add_argument("--collar", type=float, default=0)
    parser.add_argument("--skip-overlap", action="store_true")
    parser.add_argument(
        "--experiment", action="store_true", help="Compare base.en/small.en and known speaker count"
    )
    args = parser.parse_args()
    configure_logging()
    config = Config()
    config = replace(
        config,
        whisper_model=args.model or config.whisper_model,
        num_speakers=args.speakers,
        initial_prompt=args.prompt,
    )
    try:
        refs = load_ami(args.annotations, args.meeting) if args.annotations else None
        if args.experiment and (not refs or args.asr_only):
            raise ValueError("Experiments require annotations and full diarization")
        variants = [("baseline", config)]
        if args.experiment:
            variants = [
                ("baseline", replace(config, whisper_model="base.en", num_speakers=None)),
                ("larger_asr", replace(config, whisper_model="small.en", num_speakers=None)),
                (
                    "known_speakers",
                    replace(config, whisper_model="small.en", num_speakers=args.speakers or 4),
                ),
            ]
        results = []
        for name, variant in variants:
            output = args.output / name if args.experiment else args.output
            transcript = run(
                args.audio.expanduser(), variant, args.start, args.duration, args.asr_only
            )
            save(transcript, output)
            if refs:
                metrics = evaluate(
                    transcript, *refs, collar=args.collar, skip_overlap=args.skip_overlap
                )
                if args.asr_only:
                    metrics["der"] = None
                    metrics["der_components_seconds"] = None
                (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
                results.append({"variant": name, **metrics})
        if args.experiment:
            (args.output / "experiments.json").write_text(json.dumps(results, indent=2))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Pipeline failed: {error}\n")


if __name__ == "__main__":
    main()
