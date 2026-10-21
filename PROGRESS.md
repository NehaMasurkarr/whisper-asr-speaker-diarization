# Progress

## Completed milestones

- Modular `src/meeting_asr` package: WAV/FLAC loading, resampling, Whisper word ASR, Community-1 diarization, overlap-aware attribution, JSON/TXT exports.
- AMI v1.6.2 words and transcriber speech-segment parsing; WER and DER with explicit crop/UEM, collar, and overlap policy.
- CLI, experiment runner, environment configuration, credential redaction, JSON logging, MPS acceleration with CPU fallback and failed-allocation cleanup.
- Streamlit upload/playback, speaker timeline, readable transcripts, downloads, and evaluation displays.
- 19 local tests passed; Ruff lint/format and pip dependency checks passed.
- Linux GitHub Actions passed on Python 3.11 and 3.13; subsequent milestone checks monitored.
- Repository configured and tested milestones pushed using Conventional Commits.
- README with architecture, setup, methodology, limitations, and measured crop results; exact Mac dependency snapshot.
- Audio verified at user's supplied Downloads path. Extracted annotations found at `/Users/neha/Downloads/ami_public_manual_1.6.2`; all eight ES2002a word/segment XMLs match the official archive byte for byte.
- Hugging Face access resolved. PyTorch restricted-loading compatibility fixed with a scoped allowlist of three known pyannote metadata classes; unrestricted loading remains disabled.
- Real full CLI crop (60–120 seconds): tiny.en WER 22.9167%; Community-1 DER 13.1396%, zero collar, overlap included.
- ASR experiments on identical crops: base.en did not improve over tiny.en (60–120 seconds: 23.6111% versus 22.9167% WER).
- Full-meeting base.en ASR completed: 1,272.64 seconds; 36.3706% WER; 280 substitutions, 536 deletions, 138 insertions; 2,623 normalized reference words.

## Current work

- Full-meeting Community-1 diarization running on MPS with batch size 4; completed ASR is preserved for attribution.
- Combine real stage outputs and evaluate full-meeting DER against the user's AMI speech-segment references.
- Finish diarization experiment, final validation, documentation, and GitHub synchronization.

## Current blockers

- No authentication blocker remains.
- Upstream TorchCodec binary is incompatible with the pinned PyTorch binary. The implemented SoundFile/SciPy waveform path bypasses it and real pyannote inference succeeds; upstream pyannote file-path decoding is not supported in this environment.

## Operational findings

- Automatic Whisper MPS inference hit an unsupported runtime operation and successfully fell back to CPU on real audio.
- Concurrent full-meeting jobs caused memory pressure on the 8 GB host. Redundant computation was stopped; remaining inference runs sequentially.
- Initial CI missed CPU PyTorch; dependency configuration was fixed and both Python jobs passed.
- Audio, annotations, model weights, credentials, and generated outputs remain excluded from Git.
- Final engineering validation: 20 tests passed; Ruff lint/format and pip check passed.
- Added a configurable four-thread CPU default, clear invalid-environment errors, finite crop validation, and consistent speaker colors in the dashboard.
- Optional pyannote usage telemetry now defaults to disabled without overriding an explicit environment preference.
- Full diarization remains active in neural-network convolution work; CPU use and stable memory were verified with a brief process profile.

- MPS crop inference succeeded at batch size 4 and exactly matched CPU DER (13.1396%). Automatic accelerator selection is now the default, with explicit device overrides preserved.
- Upstream batches of 32 caused GPU memory pressure. Reduced batches to 4 for segmentation and embedding; added validation and adapter tests. All 20 tests and Ruff checks passed again.
- Replaced the long full-meeting CPU diarization attempt with verified MPS inference; no incomplete CPU score is reported.
- Full MPS inference completed. After clipping padded model timestamps to the real audio duration, output contains 353 turns, four speakers, 2,226 ASR word records, 210 overlap-marked words, and 40 unattributed words.
- Full-meeting metrics: WER 36.3706%; DER 27.7472% (zero collar, overlap included, AMI v1.6.2 transcriber segments).
- Controlled crop DER experiments: automatic count 13.1396%; known four speakers 10.3109%; full-meeting context 8.9958%, with ASR words and scoring region held fixed.
- Added audio-bound clipping regression coverage and exact Hugging Face revision configuration. All 20 tests passed again; loading the pinned revision was verified on a synthetic waveform.
