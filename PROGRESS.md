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
