import streamlit as st
from predictor import prever_k1, prever_k2

st.set_page_config(page_title="Predição k1 e k2", layout="centered")

st.title("Determinação das rigidezes k1 e k2")
st.write("Modelos baseados em Redes Neurais Artificiais treinadas previamente.")

tab_k2, tab_k1 = st.tabs(["k2", "k1"])

# =========================
# TAB K2
# =========================
with tab_k2:
    st.subheader("Cálculo da rigidez k2")

    d = st.number_input("Altura do perfil d (mm)", min_value=0.0, value=600.0)
    p = st.number_input("Razão p / a₀", min_value=0.0, value=1.5)
    tw = st.number_input("Espessura da alma tw (mm)", min_value=0.0, value=12.5)
    a0 = st.number_input("Diâmetro da abertura a₀ (mm)", min_value=0.0, value=360.0)

    if st.button("Calcular (k2)"):
        with st.spinner("Calculando k2..."):
            df_result = prever_k2(d, p, tw, a0)

        st.success("Predição de k2 concluída com sucesso!")
        st.dataframe(df_result)

        csv = df_result.to_csv(sep=';', index=False).encode('utf-8')
        st.download_button(
            label="Baixar resultados k2 (CSV)",
            data=csv,
            file_name="resultados_k2.csv",
            mime="text/csv"
        )

# =========================
# TAB K1
# =========================
with tab_k1:
    st.subheader("Cálculo da rigidez k1")

    fc = st.number_input("Resistência do concreto fc (MPa)", min_value=0.0, value=50.0)
    hs = st.number_input("Espessura da laje hs (mm)", min_value=0.0, value=80.0)
    a  = st.number_input("Distância entre vigas de aço paralelas a (mm)", min_value=0.0, value=1500.0)
    As = st.number_input("Área de armadura negativa As (mm²)", min_value=0.0, value=301.59, format="%.2f")

    if st.button("Calcular (k1)"):
        with st.spinner("Calculando k1..."):
            df_result = prever_k1(fc, hs, a, As)

        st.success("Predição de k1 concluída com sucesso!")
        st.dataframe(df_result)

        csv = df_result.to_csv(sep=';', index=False).encode('utf-8')
        st.download_button(
            label="Baixar resultados k1 (CSV)",
            data=csv,
            file_name="resultados_k1.csv",
            mime="text/csv"
        )
