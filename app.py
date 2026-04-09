import streamlit as st
import os
from model import train_model, load_model, predict, MODEL_PATH

st.title("🌽 Maize Leaf Detection (Stable Version)")

# Train
st.header("Train Model")

train_files = st.file_uploader(
    "Upload training images (healthy_ or blight_)",
    accept_multiple_files=True
)

if st.button("Train"):
    if not train_files:
        st.warning("Upload images first")
    else:
        with st.spinner("Training..."):
            model = train_model(train_files)

        st.success("Training complete!")

# Load
if os.path.exists(MODEL_PATH):
    model = load_model()
    st.success("Model ready!")
else:
    model = None

# Predict
st.header("Predict")

file = st.file_uploader("Upload image to predict")

if file:
    st.image(file)

    if file.size > 2 * 1024 * 1024:
        st.error("Image too large (max 2MB)")
    elif model:
        result = predict(model, file)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Train model first")
