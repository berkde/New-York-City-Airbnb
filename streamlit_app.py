import joblib
import numpy as np
import streamlit as st

model = joblib.load("model.pkl")
X_train_final = joblib.load('X_train_final.pkl')
df = X_train_final

st.title("NYC Host Popularity Predictor")

row_idx = st.selectbox("Select a property row to auto-fill features:", df.index)
selected_row = df.loc[row_idx]

features = X_train_final.columns.tolist()
input_data = selected_row[features].to_frame().T

st.write("Selected Airbnb Property Details:", input_data)

if st.button("Predict Popularity"):
    pred_log = model.predict(input_data)
    popularity = np.expm1(pred_log[0])
    st.success(f"📈 Estimated Host Popularity: {popularity:.3f}")
    st.caption("Note: Popularity ranges from 0 (low) to 1 (high)")
