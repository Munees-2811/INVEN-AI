"""AI chatbot page — conversational, grounded in live business data."""
from __future__ import annotations

import streamlit as st

from src.ai import claude_client
from src.ai.chatbot import answer, build_context
from ui.state import data_version, get_data, get_health


def render() -> None:
    st.header("💬 AI Inventory Assistant")
    if claude_client.available():
        st.caption("🟢 Connected to Claude — full conversational AI enabled.")
    else:
        st.caption("🟡 Offline mode — set `ANTHROPIC_API_KEY` for full conversational AI. "
                   "Rule-based answers are active.")

    v = data_version()
    data = get_data(v)
    health = get_health(v)
    ctx = build_context(data, health)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    examples = ["What should I reorder?", "How are my sales trending?",
                "Which products are overstocked?", "Show my top sellers"]
    cols = st.columns(len(examples))
    clicked = None
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            clicked = ex

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about inventory, sales, suppliers…") or clicked
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = answer(prompt, ctx, history=st.session_state.chat_history[:-1])
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.session_state.chat_history and st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()
