"""AI product categorization page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.models.categorization import suggest_category, train_categorizer
from ui.state import data_version, get_data


@st.cache_resource(show_spinner="Training categorizer…")
def _model(_v: int):
    data = get_data(_v)
    return train_categorizer(data["products"])


def render() -> None:
    st.header("🏷️ AI Product Categorization")
    v = data_version()
    model = _model(v)

    acc = model.cv_accuracy
    st.metric("Cross-validated accuracy", f"{acc*100:.1f}%" if acc == acc else "n/a",
              help="Char n-gram TF-IDF + Logistic Regression over product names")
    st.caption(f"Known categories: {', '.join(model.classes)}")

    st.divider()
    st.subheader("Categorize a new product")
    name = st.text_input("Product name", placeholder="e.g. Sparkling Water 750ml")
    if name:
        res = suggest_category(model, name)
        c1, c2 = st.columns(2)
        c1.metric("Predicted category", res["predicted_category"])
        c2.metric("Confidence", f"{res['confidence']*100:.0f}%")

    st.divider()
    st.subheader("Bulk categorize")
    txt = st.text_area("One product name per line",
                       placeholder="Almond Milk 1L\nDark Chocolate 90%\nGlass Cleaner Spray")
    if st.button("Categorize all") and txt.strip():
        names = [n.strip() for n in txt.splitlines() if n.strip()]
        preds = model.predict(names)
        st.dataframe(pd.DataFrame(preds), use_container_width=True, hide_index=True)
