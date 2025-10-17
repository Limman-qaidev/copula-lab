from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import streamlit as st
from streamlit.navigation.page import StreamlitPage

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

st.set_page_config(
    page_title="Copula Lab",
    page_icon="📈",
    layout="wide",
)

DATA_PAGE: StreamlitPage = st.Page(
    "pages/1_Data.py",
    title="Data",
    icon="📄",
    url_path="data",
)
CALIBRATE_PAGE: StreamlitPage = st.Page(
    "pages/2_Calibrate.py",
    title="Calibrate",
    icon="⚙️",
    url_path="calibrate",
)
COMPARE_PAGE: StreamlitPage = st.Page(
    "pages/3_Compare.py",
    title="Compare",
    icon="📊",
    url_path="compare",
)
STUDY_PAGE: StreamlitPage = st.Page(
    "pages/4_Study.py",
    title="Study",
    icon="📚",
    url_path="study",
)
SANDBOX_PAGE: StreamlitPage = st.Page(
    "pages/5_Sandbox.py",
    title="Sandbox",
    icon="🧪",
    url_path="sandbox",
)


def render_home() -> None:
    st.title("Home")
    st.caption("Interactive laboratory to study and calibrate copulas.")

    hero_left, hero_right = st.columns((2, 1))

    with hero_left:
        st.markdown(
            """
### Welcome to Copula Lab

Upload your data, calibrate copula models, and visualize the
results with tooling crafted for quantitative practitioners.
Follow the recommended flow or explore each page independently.
"""
        )

        st.markdown(
            """
**Suggested flow**
1. **Data**: upload your dataset, choose the relevant columns,
   and build the PIT.
2. **Calibrate**: fit parametric copulas with metrics and logging.
3. **Compare**: contrast candidates and inspect diagnostic charts.
4. **Study**: consult notes, formulas, and reference material.
"""
        )

    with hero_right:
        st.markdown("#### Quick status")
        st.metric("Supported dimensionality", "Multivariate")
        st.metric("Default seed", "42")
        st.info(
            "Head to *Data* to start, or use the links below to jump to any "
            "module."
        )

    st.divider()

    cards = st.columns(3)
    card_specs = [
        (
            "📄 Data",
            (
                "Load multivariate CSVs and work with "
                "pseudo-observations in (0, 1)."
            ),
            DATA_PAGE,
        ),
        (
            "⚙️ Calibrate",
            "Estimate parameters, manage seeds, and monitor convergence.",
            CALIBRATE_PAGE,
        ),
        (
            "📊 Compare",
            "Contrast copulas, inspect tail dependence, and quantify risks.",
            COMPARE_PAGE,
        ),
    ]

    for column, (title, description, target_page) in zip(cards, card_specs):
        with column:
            st.subheader(title)
            st.write(description)
            st.page_link(target_page, label="Open", icon="➡️")

    st.page_link(STUDY_PAGE, label="Open Study", icon="📘")
    st.page_link(SANDBOX_PAGE, label="Explore Sandbox", icon="🧪")

    st.divider()

    resources, session_panel = st.columns((2, 1))

    with resources:
        st.markdown(
            """
### Quick resources
- Technical documentation in `docs/` with theory and worked examples.
- Reproducible experiments inside `notebooks/`.
- Utility scripts under `src/utils/` ready to be reused.
"""
        )

    with session_panel:
        st.markdown("### Session status")
        if "U" in st.session_state:
            U = st.session_state["U"]
            st.success(
                "Pseudo-observations available: "
                f"n={U.shape[0]}, d={U.shape[1]}"
            )
        else:
            st.info("Pseudo-observations have not been generated yet.")


HOME_PAGE: StreamlitPage = st.Page(
    render_home, title="Home", icon="📈", default=True
)

PAGES: Sequence[StreamlitPage] = [
    HOME_PAGE,
    DATA_PAGE,
    CALIBRATE_PAGE,
    COMPARE_PAGE,
    STUDY_PAGE,
    SANDBOX_PAGE,
]

selected_page: StreamlitPage = st.navigation(PAGES, position="sidebar")
selected_page.run()
