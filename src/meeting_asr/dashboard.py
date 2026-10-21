"""Run with streamlit run src/meeting_asr/dashboard.py."""

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.colors import qualitative
import streamlit as st
from meeting_asr.config import Config
from meeting_asr.pipeline import run, readable
from meeting_asr.evaluation import load_ami, evaluate

load_dotenv()
st.set_page_config(page_title="Meeting Speech Lab", layout="wide")
st.title("Meeting Speech Lab")
st.caption("Local Whisper transcription • pyannote speaker activity • AMI evaluation")
upload = st.file_uploader("Upload meeting audio", type=["wav", "flac"])
if upload is not None:
    st.audio(upload)
model = st.selectbox("Whisper model", ["base.en", "small.en", "medium.en", "tiny.en"])
duration = st.number_input("Preview duration in seconds", min_value=1, max_value=7200, value=60)
speakers = st.number_input("Known speakers (0 = automatic)", min_value=0, max_value=30, value=0)
asr_only = st.checkbox("ASR only (no Hugging Face token required)")
annotations = st.text_input("AMI annotation directory (words and segments) (optional, local path)")
meeting = st.text_input("AMI meeting identifier", value="ES2002a")
st.info(
    "Audio stays on the machine hosting this app. Models may download on first use. "
    "Use only on a trusted local host; inference can take several minutes."
)
if st.button("Transcribe", disabled=upload is None):
    st.session_state.pop("transcript", None)
    st.session_state.pop("metrics", None)
    try:
        with TemporaryDirectory() as folder, st.spinner("Running models…"):
            path = Path(folder) / "upload.wav"
            path.write_bytes(upload.getvalue())
            transcript = run(
                path,
                Config(whisper_model=model, num_speakers=speakers or None),
                duration=float(duration),
                asr_only=asr_only,
            )
            transcript.metadata["audio_name"] = Path(upload.name).name
            st.session_state.transcript = transcript
            if annotations:
                try:
                    metrics = evaluate(
                        transcript, *load_ami(Path(annotations).expanduser(), meeting)
                    )
                    if asr_only:
                        metrics["der"] = None
                        metrics["der_components_seconds"] = None
                    st.session_state.metrics = metrics
                except (OSError, ValueError, ImportError) as error:
                    st.warning(f"Transcript is ready; evaluation unavailable: {error}")
    except Exception as error:
        st.error(f"Processing failed: {error}")
if "transcript" in st.session_state:
    transcript = st.session_state.transcript
    fig = go.Figure()
    palette = qualitative.Safe
    colors = {
        speaker: palette[i % len(palette)]
        for i, speaker in enumerate(sorted({t.speaker for t in transcript.turns}))
    }
    for turn in transcript.turns:
        fig.add_trace(
            go.Scatter(
                x=[turn.start, turn.end],
                y=[turn.speaker] * 2,
                mode="lines",
                line={"width": 14, "color": colors[turn.speaker]},
                name=turn.speaker,
                hovertemplate="%{y}<br>%{x:.2f}s<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(xaxis_title="Seconds from crop start", yaxis_title="Speaker")
    st.plotly_chart(fig, use_container_width=True)
    st.text(readable(transcript))
    st.download_button("Download JSON", json.dumps(asdict(transcript), indent=2), "transcript.json")
    st.download_button("Download TXT", readable(transcript), "transcript.txt")
    if "metrics" in st.session_state:
        metrics = st.session_state.metrics
        wer_column, der_column = st.columns(2)
        wer_column.metric("WER", f"{metrics['wer']:.2%}")
        der_column.metric(
            "DER", "Unavailable" if metrics["der"] is None else f"{metrics['der']:.2%}"
        )
        st.json(metrics)
    else:
        st.caption("Evaluation unavailable: supply matching AMI annotations to calculate metrics.")
