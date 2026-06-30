"""OCR invoice scanning page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ai import claude_client
from src.ai.ocr import scan_invoice


def render() -> None:
    st.header("🧾 OCR Invoice Scanning")
    if claude_client.available():
        st.caption("🟢 Using Claude vision for structured invoice extraction.")
    else:
        st.caption("🟡 Claude not configured — using local OCR fallback (requires Tesseract installed).")

    up = st.file_uploader("Upload a supplier invoice image", type=["png", "jpg", "jpeg", "webp"])
    if up is None:
        st.info("Upload an invoice photo or scan to extract line items automatically.")
        return

    media = f"image/{'jpeg' if up.type.endswith(('jpg', 'jpeg')) else up.type.split('/')[-1]}"
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(up, caption="Uploaded invoice", use_container_width=True)
    with c2:
        with st.spinner("Reading invoice…"):
            result = scan_invoice(up.getvalue(), media_type=media)

        st.caption(f"Engine: `{result.get('_engine', 'unknown')}`")
        if result.get("_error"):
            st.error(result["_error"])

        meta = {k: result.get(k) for k in
                ("supplier_name", "invoice_number", "invoice_date", "grand_total") if result.get(k)}
        if meta:
            st.json(meta)

        items = result.get("line_items") or []
        if items:
            df = pd.DataFrame(items)
            st.subheader("Extracted line items")
            edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", hide_index=True)
            st.download_button("⬇️ Download as CSV", edited.to_csv(index=False),
                               "invoice_items.csv", "text/csv")
        elif result.get("raw_text"):
            st.text_area("Raw OCR text", result["raw_text"], height=240)
        else:
            st.warning("No line items could be extracted.")
