# Progress

## Status

Implementation, real-data inference, evaluation, experiments, local verification, and documentation are complete. Tested milestones have been pushed with Conventional Commits. Final GitHub checks are verified during publication.

## Completed milestones

- Modular, type-hinted `src/meeting_asr` package with environment configuration and JSON logging.
- OpenAI Whisper word ASR; Community-1 overlapping speaker diarization; timestamp alignment and speaker-attributed JSON/TXT exports.
- Apple Silicon support: verified Whisper CPU fallback and real MPS diarization. Configurable CPU threads and bounded segmentation/embedding batches default to four.
- Graceful missing-reference and model-access errors; token redaction; ignored/private `.env`; optional telemetry disabled by default.
- AMI word and transcriber speech-segment parsing; normalized WER; optimal speaker-mapped DER with recorded scoring region, collar, and overlap policy.
- CLI, accuracy experiment runner, and reusable Python API for future interviewer integration.
- Streamlit audio upload/playback, consistent speaker timeline, readable transcript, downloads, and metric displays.
- Reproducible dependency snapshot, pinned model-revision support, architecture/setup/methodology documentation, and GitHub Actions CI.
- Regression coverage for speaker changes, overlaps, gaps, audio resampling/crops, model adapters, authentication, checkpoint loading, timestamp padding, metrics, CLI, and dashboard.

## Verified real-data results

Audio: `~/Downloads/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav`.
References: `~/Downloads/ami_public_manual_1.6.2`, verified byte-for-byte against the official AMI archive for all eight ES2002a words/segments files.

- Complete duration: 1,272.64 seconds.
- ASR: base.en on CPU after MPS fallback; diarization: Community-1 on MPS, batch size 4.
- Full-meeting WER: **36.3706%** (280 substitutions, 536 deletions, 138 insertions; 2,623 normalized reference words).
- Full-meeting DER: **27.7472%**, zero collar, overlap included, against AMI transcriber speech segments.
- Four speakers, 353 bounded turns, 2,226 ASR word records, 210 overlap-marked words, 40 unattributed words.
- Local deliverables: `outputs/ES2002a-full/transcript.json`, `transcript.txt`, and `metrics.json`.
- Artifact bounds, source SHA-256, scoring duration, WER arithmetic, overlap/unknown-speaker rendering, and actual device metadata verified.
- Model revision: `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`; loading this exact revision was also smoke-tested.

## Accuracy experiments

On seconds 60–120, using the same 144 normalized reference words:

- ASR: tiny.en WER 22.9167%; base.en WER 23.6111%. Larger ASR did not improve this crop.
- Diarization, holding ASR words fixed: automatic count DER 13.1396%; known four speakers DER 10.3109%; full-meeting context DER 8.9958%.
- Known speaker count uses disclosed prior information; full context uses the whole recording before scoring the crop. These are single-meeting observations, not corpus-level benchmark claims.
- CPU/batch-32 and MPS/batch-4 automatic-count crop DER matched exactly; no controlled speed claim is made.

## Verification

- `pytest -q`: **20 passed** after the final code changes.
- Ruff lint and formatting checks passed.
- `pip check`: no broken requirements.
- GitHub Actions matrix covers Python 3.11 and 3.13; code milestones pass without tokens or model downloads.
- No audio, annotations, model weights, credentials, or generated transcripts are tracked by Git.

## Resolved issues

- GitHub authentication repaired; regular tested commits pushed to the existing public repository.
- Hugging Face gated access granted; restricted PyTorch checkpoint loading fixed using a scoped allowlist of the three known metadata classes.
- CPU inference contention and upstream batch-32 GPU memory pressure addressed with sequential full inference and batch size 4.
- Full diarization switched to verified MPS after the short GPU test succeeded; completed ASR was reused.
- Padded pyannote turns clipped to audio bounds; one wholly out-of-bounds turn discarded. Metrics were unchanged because scoring already used the exact audio region.
- Initial CI dependency omission fixed by installing CPU PyTorch and dashboard dependencies.

## Blockers and limitations

No blocker remains for the supported WAV/FLAC waveform pipeline. The optional upstream TorchCodec decoder is incompatible with the pinned PyTorch binary; SoundFile/SciPy waveform loading bypasses it and has completed real full-meeting inference. Direct upstream file-path decoding is outside the validated path.

Whisper may miss simultaneous words; overlap detection does not separate voices. Speaker IDs are session-local. Results cover one headset-mix meeting, not a held-out AMI benchmark or a production-quality guarantee.

## Optional next steps

Evaluate additional held-out meetings, expand accuracy/model experiments, and build streaming/session orchestration for the future AI Interviewer using the existing module boundaries.
