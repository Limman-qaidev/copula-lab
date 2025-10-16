import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

st.set_page_config(
    page_title="Copula Lab · Home",
    page_icon="📈",
    layout="wide",
)
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
card_texts = [
    (
        "📄 Data",
        "Load multivariate CSVs and work with pseudo-observations in (0, 1).",
        "pages/1_Data.py",
    ),
    (
        "⚙️ Calibrate",
        "Estimate parameters, manage seeds, and monitor convergence.",
        "pages/2_Calibrate.py",
    ),
    (
        "📊 Compare",
        "Contrast copulas, inspect tail dependence, and quantify risks.",
        "pages/3_Compare.py",
    ),
]

for column, (title, description, target) in zip(cards, card_texts):
    with column:
        st.subheader(title)
        st.write(description)
        st.page_link(target, label="Open", icon="➡️")

st.page_link("pages/4_Study.py", label="📚 Open Study", icon="📘")
st.page_link("pages/5_Sandbox.py", label="🧪 Explore Sandbox", icon="🧪")

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
            f"Pseudo-observations available: n={U.shape[0]}, d={U.shape[1]}"
        )
    else:
        st.info("Pseudo-observations have not been generated yet.")
