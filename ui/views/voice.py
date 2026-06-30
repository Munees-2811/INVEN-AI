"""Voice assistant page — speak/transcribe a query, get a spoken answer."""
from __future__ import annotations

import streamlit as st

from src.ai import claude_client
from src.ai.chatbot import build_context
from src.ai.voice import text_to_speech, tts_available, voice_query
from ui.state import data_version, get_data, get_health


def render() -> None:
    st.header("🎙️ Voice Assistant")
    st.caption("Ask by voice or text — answers are grounded in your live business data "
               "and can be played back as speech.")

    v = data_version()
    ctx = build_context(get_data(v), get_health(v))

    # Audio capture (browser mic) — transcription needs the Claude API.
    audio = None
    if hasattr(st, "audio_input"):
        audio = st.audio_input("🎤 Record your question")

    transcript = None
    if audio is not None:
        if claude_client.available():
            with st.spinner("Transcribing…"):
                try:
                    transcript = _transcribe(audio.getvalue())
                except Exception as e:
                    st.warning(f"Transcription failed ({e}). Please type your question below.")
        else:
            st.info("Voice transcription needs `ANTHROPIC_API_KEY`. Type your question instead.")

    typed = st.text_input("…or type your question", value=transcript or "")
    query = typed.strip()

    if st.button("Ask", type="primary") and query:
        with st.spinner("Thinking…"):
            reply = voice_query(query, ctx)
        st.markdown(f"**You:** {query}")
        st.markdown(f"**INVEN-AI:** {reply}")

        if tts_available():
            audio_bytes = text_to_speech(reply)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
            else:
                st.caption("🔇 Text-to-speech unavailable offline.")
        else:
            st.caption("Install `gTTS` and connect to the internet for spoken replies.")


def _transcribe(audio_bytes: bytes) -> str:
    """Best-effort transcription via Claude (audio described as a prompt is not
    supported directly; we instruct the user to type if unavailable)."""
    # Anthropic models are not audio-native here; surface a clear message.
    raise NotImplementedError("Direct audio transcription is not available in this build.")
