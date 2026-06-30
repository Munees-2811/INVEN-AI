"""
Voice assistant helpers.

Text-to-speech via gTTS (returns MP3 bytes for st.audio). Speech-to-text is
handled at the UI layer through an uploaded audio clip; here we expose a helper
that turns a transcript into a spoken, grounded answer by reusing the chatbot.
"""
from __future__ import annotations

import io


def text_to_speech(text: str, lang: str = "en") -> bytes | None:
    """Return MP3 audio bytes for ``text``; None if gTTS unavailable/offline."""
    try:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text[:1500], lang=lang).write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def voice_query(transcript: str, context: dict, history: list[dict] | None = None) -> str:
    """Answer a spoken query using the same grounded chatbot brain."""
    from src.ai.chatbot import answer

    return answer(transcript, context, history=history)


def tts_available() -> bool:
    try:
        import gtts  # noqa: F401

        return True
    except Exception:
        return False
