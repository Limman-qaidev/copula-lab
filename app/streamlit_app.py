import streamlit as st

st.set_page_config(page_title="Copula Lab", layout="wide")
st.title("Copula Lab")

st.markdown(
    """
Calibrate and compare **copulas** on financial data with live diagnostics.

**Quick start**
1. Go to **Data** to upload a CSV/Parquet and select columns.
2. Go to **Calibrate** to fit a copula (placeholder in this step).
3. Use **Compare** and **Study** to explore models and theory.
"""
)

# Handy links to pages
st.page_link("pages/1_Data.py", label="→ Data", icon="📄")
st.page_link("pages/2_Calibrate.py", label="→ Calibrate", icon="⚙️")
st.page_link("pages/3_Compare.py", label="→ Compare", icon="📊")
st.page_link("pages/4_Study.py", label="→ Study", icon="📚")
st.page_link("pages/5_Sandbox.py", label="→ Sandbox", icon="🧪")

with st.sidebar:
    st.header("Session")
    df = st.session_state.get("data_df")
    if df is None:
        st.info("No dataset loaded.")
    else:
        st.success(
            f"Dataset in session: {df.shape[0]} rows × {df.shape[1]} cols"
        )
