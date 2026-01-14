import streamlit as st
from predictor import prever_k2

st.set_page_config(page_title="Predição de k2", layout="centered")

st.title("Predição da rigidez k2")
st.write("Modelo baseado em Rede Neural Artificial treinada previamente.")

# ===== Inputs =====
d = st.number_input("Altura do perfil d (mm)", min_value=0.0, value=600.0)
p = st.number_input("Razão p / a₀", min_value=0.0, value=1.5)
tw = st.number_input("Espessura da alma tw (mm)", min_value=0.0, value=12.5)
a0 = st.number_input("Altura da abertura a₀ (mm)", min_value=0.0, value=360.0)

# ===== Botão =====
if st.button("Rodar predição"):
    with st.spinner("Calculando k2..."):
        df_result = prever_k2(d, p, tw, a0)

    st.success("Predição concluída com sucesso!")

    st.dataframe(df_result)

    # Download
    csv = df_result.to_csv(sep=';', index=False).encode('utf-8')

    st.download_button(
        label="Baixar resultados (CSV)",
        data=csv,
        file_name="resultados_k2.csv",
        mime="text/csv"
    )
