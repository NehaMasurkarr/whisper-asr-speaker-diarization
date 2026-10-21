# Progress

## Completed
- Inspected empty repository, existing GitHub remote, and local ES2002a audio.
- Confirmed GitHub CLI authentication is invalid; no local AMI annotations found.

## Current work
- Implement modular inference, overlap-aware alignment, reference evaluation, CLI, and dashboard.

## Blockers
- GitHub push requires `gh auth login -h github.com`.
- Full diarization requires a Hugging Face token and acceptance of the Community-1 model terms.
- Real evaluation requires AMI word annotations and an explicit scoring region.

## Next steps
- Run unit/integration tests and Ruff; install and smoke-test real inference dependencies.
- Commit tested milestones and push when authentication is available.

## Validation update
- Full dependency installation succeeded; `pip check` reports no broken requirements.
- Initial suite: 11 tests passed, including dashboard AppTest and real pyannote.metrics scoring.
- Ruff lint and format checks passed; two additional annotation/audio edge-case tests added.
- Official AMI v1.6.2 references downloaded locally: 2,600 timed words and 277 transcriber speech segments for ES2002a.
- DER now defaults to transcriber speech segments, rather than lexical-word activity.
- GitHub network authentication verified; public repository already exists.
- Real Whisper tiny.en CPU smoke/evaluation is running on seconds 30–60.
- Optional TorchCodec decoder has a binary incompatibility with the selected PyTorch build. The implemented waveform input bypasses it; file-path decoding through upstream pyannote is not supported by this environment.
- Expanded suite: 13 passed; Ruff lint/format passed.
- Real CPU tiny.en ASR run completed on ES2002a seconds 30–60; JSON/TXT and evaluation artifacts generated under ignored outputs/.
- Measured crop WER: 1.0714285714 (107.14%), 14 reference words, 3 substitutions, 2 deletions, 10 insertions. WER can exceed 100% due to insertions. This short crop is a smoke result, not corpus-wide quality.
- DER is null for ASR-only output; no diarization score has been invented.

## Integration follow-up
- Full real diarization attempt reached Hugging Face but returned HTTP 403 for Community-1. Token is configured; model access conditions or gated-repository permissions still need correction.
- Real tiny.en automatic-device run exercised an MPS runtime failure and successful CPU fallback.
- base.en on the same 30–60 second crop measured WER 1.2142857143 (121.43%): larger model did not improve this sample.
- Local suite expanded to 17 passing tests; JSON log escaping and credential-error redaction verified.
- Initial Linux CI exposed a missing test dependency (PyTorch). Updated CI to install CPU PyTorch and dashboard dependencies; rerun pending.
- Corrected Linux CI passed on Python 3.11 and 3.13 (run 39533266778).
- Dashboard includes audio playback, metric cards, and transcript preservation when evaluation inputs are unavailable; 18 local tests now pass.
- Found user's extracted AMI references in Downloads. All eight ES2002a words/segments XML files match the official archive byte for byte.
- Full-meeting ASR and full-pipeline runs launched against the user's dataset. Updated model-access retry still failed; diagnosing the underlying exception without revealing credentials.
- Underlying model failure identified as PyTorch restricted checkpoint deserialization, after Hugging Face access was fixed.
- Added a scoped allowlist for the three pyannote metadata classes actually present in the official segmentation and embedding checkpoints; unrestricted pickle loading remains disabled.
- Added a regression test that loads custom checkpoint metadata only inside the scope and verifies cleanup afterward.
- Stopped a redundant full-ASR process; retaining the original full-meeting ASR run for reuse with full diarization.
- Real end-to-end CLI check succeeded on ES2002a seconds 60–120: tiny.en WER 22.9167%, Community-1 DER 13.1396% (zero collar, overlap included; AMI transcriber speech segments).
- Full pipeline sample produced speaker-attributed JSON/TXT and actual evaluation metrics.
- Host has 8 GB RAM; concurrent full-meeting jobs caused memory pressure. Stopped full diarization until ASR completes and switched to sequential inference.
- Improved MPS fallback to exit the exception scope and release failed GPU tensors before loading the CPU model. Adapter regression tests: 7 passed.
