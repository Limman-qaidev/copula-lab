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
st.caption("Laboratorio interactivo para estudiar y calibrar cópulas.")

hero_left, hero_right = st.columns((2, 1))

with hero_left:
    st.markdown(
        """
### Bienvenido a Copula Lab

Pon tus datos en contexto, calibra modelos copulares y visualiza los
resultados con herramientas diseñadas para analistas cuantitativos.
Sigue el flujo recomendado o explora libremente las páginas.
"""
    )

    st.markdown(
        """
**Flujo sugerido**
1. **Data**: carga tu dataset, define las columnas relevantes y genera el PIT.
2. **Calibrate**: ajusta cópulas paramétricas con métricas y logs.
3. **Compare**: enfrenta alternativas y revisa gráficos diagnósticos.
4. **Study**: accede a notas, fórmulas y enlaces de referencia.
"""
    )

with hero_right:
    st.markdown("#### Estado rápido")
    st.metric("Dimensión soportada", "Multivariante")
    st.metric("Semilla por defecto", "42")
    st.info(
        "Accede a *Data* para comenzar o utiliza los enlaces inferiores "
        "para saltar directamente a cualquier módulo."
    )

st.divider()

cards = st.columns(3)
card_texts = [
    (
        "📄 Datos",
        (
            "Carga CSVs multivariados y trabaja con pseudo-observaciones "
            "en (0,1)."
        ),
        "pages/1_Data.py",
    ),
    (
        "⚙️ Calibración",
        "Estima parámetros, controla semillas y supervisa convergencia.",
        "pages/2_Calibrate.py",
    ),
    (
        "📊 Comparativa",
        "Contrasta copulas, examina dependencias tail y cuantifica riesgos.",
        "pages/3_Compare.py",
    ),
]

for column, (title, description, target) in zip(cards, card_texts):
    with column:
        st.subheader(title)
        st.write(description)
        st.page_link(target, label="Abrir", icon="➡️")

st.page_link("pages/4_Study.py", label="📚 Accede a Study", icon="📘")
st.page_link("pages/5_Sandbox.py", label="🧪 Explorar Sandbox", icon="🧪")

st.divider()

resources, session_panel = st.columns((2, 1))

with resources:
    st.markdown(
        """
### Recursos rápidos
- Documentación técnica en `docs/` con teoría y ejemplos.
- Cuadernos en `notebooks/` para experimentos reproducibles.
- Scripts utilitarios en `src/utils/` listos para reutilizar.
"""
    )

with session_panel:
    st.markdown("### Estado de la sesión")
    if "U" in st.session_state:
        U = st.session_state["U"]
        st.success(
            f"Pseudo-observaciones disponibles: n={U.shape[0]}, d={U.shape[1]}"
        )
    else:
        st.info("Aún no se han generado pseudo-observaciones.")
