import streamlit as st
from predictor import prever_k1, prever_k2


st.set_page_config(page_title="k1 and k2 Prediction", layout="centered")

st.title("Determination of Rotational Stiffness k1 and k2")
st.write("Models based on previously trained Artificial Neural Networks (ANNs).")

tab_k2, tab_k1 = st.tabs(["k2", "k1"])

# =========================
# TAB K2
# =========================
with tab_k2:
    st.subheader("Calculation of stiffness k2")

    d = st.number_input("Section depth d (mm)", min_value=0.0, value=600.0)
    p = st.number_input("Ratio p / a₀", min_value=0.0, value=1.5)
    tw = st.number_input("Web thickness tw (mm)", min_value=0.0, value=12.5)
    a0 = st.number_input("Opening diameter a₀ (mm)", min_value=0.0, value=360.0)

    if st.button("Calculate k2"):
        with st.spinner("Computing k2..."):
            df_result = predict_k2(d, p, tw, a0)

        st.success("k2 prediction completed successfully!")
        st.dataframe(df_result)

        csv = df_result.to_csv(sep=';', index=False).encode('utf-8')
        st.download_button(
            label="Download k2 results (CSV)",
            data=csv,
            file_name="k2_results.csv",
            mime="text/csv"
        )

# =========================
# TAB K1
# =========================
with tab_k1:
    st.subheader("Calculation of stiffness k1")

    fc = st.number_input("Concrete compressive strength fc (MPa)", min_value=0.0, value=50.0)
    hs = st.number_input("Slab thickness hs (mm)", min_value=0.0, value=80.0)
    a  = st.number_input("Spacing between parallel steel beams a (mm)", min_value=0.0, value=1500.0)
    As = st.number_input("Negative reinforcement area As (mm²)", min_value=0.0, value=301.59, format="%.2f")

    if st.button("Calculate k1"):
        with st.spinner("Computing k1..."):
            df_result = predict_k1(fc, hs, a, As)

        st.success("k1 prediction completed successfully!")
        st.dataframe(df_result)

        csv = df_result.to_csv(sep=';', index=False).encode('utf-8')
        st.download_button(
            label="Download k1 results (CSV)",
            data=csv,
            file_name="k1_results.csv",
            mime="text/csv"
        )
