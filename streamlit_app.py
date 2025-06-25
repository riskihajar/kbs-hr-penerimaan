import streamlit as st
import pandas as pd
import joblib

# Load model
rf_model = joblib.load('model_random_forest.pkl')
ann_model = joblib.load('model_ann.pkl')

features = [
    "AgeAtStart", "TenureMonths", "GenderCode", "RaceDesc", "MaritalDesc",
    "EmployeeType", "PayZone", "Performance Score", "BusinessUnit",
    "EmployeeClassificationType"
]

categorical_cols = [
    "GenderCode", "RaceDesc", "MaritalDesc", "EmployeeType",
    "PayZone", "Performance Score", "BusinessUnit", "EmployeeClassificationType"
]

def rule_based_accept(row):
    if row["AgeAtStart"] < 25 and row.get("Performance Score_Low", 0) == 1:
        return 0
    elif row["TenureMonths"] > 24 and row.get("Performance Score_High", 0) == 1:
        return 1
    elif row.get("EmployeeClassificationType_Intern", 0) == 1:
        return 0
    elif row.get("Performance Score_High", 0) == 1:
        return 1
    else:
        return 1

st.set_page_config(
    page_title="KBS-HR | Prediksi Penerimaan",
    page_icon="📊",
    layout="wide"
)

st.title("📊 KBS-HR: Prediksi Penerimaan Karyawan")
uploaded_file = st.file_uploader("Upload file CSV", type="csv")

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Data Awal")
    st.dataframe(raw_df)

    df_input = raw_df[features].copy()
    df_onehot = pd.get_dummies(df_input, columns=categorical_cols)

    # Siapkan salinan untuk model (sesuai kolom training)
    df_encoded = df_onehot.copy()
    for col in rf_model.feature_names_in_:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[rf_model.feature_names_in_]

    rf_pred = rf_model.predict(df_encoded)
    rf_conf = rf_model.predict_proba(df_encoded)[:, 1]
    ann_pred = ann_model.predict(df_encoded)
    ann_conf = ann_model.predict_proba(df_encoded)[:, 1]
    rule_pred = df_onehot.apply(rule_based_accept, axis=1)

    hasil_df = pd.DataFrame({
        "RF_Pred": rf_pred,
        "RF_Conf": rf_conf,
        "ANN_Pred": ann_pred,
        "ANN_Conf": ann_conf,
        "Rule_Pred": rule_pred
    })

    hasil_final = pd.concat([raw_df.reset_index(drop=True), hasil_df], axis=1)
    st.subheader("📌 Hasil Prediksi")
    st.dataframe(hasil_final)

    csv = hasil_final.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download hasil prediksi", csv, "hasil_prediksi.csv", "text/csv")
