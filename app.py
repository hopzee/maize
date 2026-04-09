# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
from model import train_model, load_trained_model, infer_label_from_filename, MODEL_SAVE_PATH

st.title("Maize Leaf Blight Detection")
st.write("Upload images to train the model automatically based on filenames, or predict new leaves.")

tab = st.radio("Choose Action", ["Train Model", "Predict Leaf"])

if tab == "Train Model":
    st.write("Upload images (1 or more) to train the model. Filename must include 'healthy' or 'blight'.")
    uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=['png','jpg','jpeg'])
    
    if uploaded_files:
        try:
            images = [Image.open(f) for f in uploaded_files]
            labels = [infer_label_from_filename(f.name) for f in uploaded_files]
        except ValueError as e:
            st.error(str(e))
        else:
            epochs = st.number_input("Epochs", min_value=1, max_value=50, value=5)
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    model = train_model(images, labels, epochs=epochs)
                st.success(f"Model trained and saved as {MODEL_SAVE_PATH}")

elif tab == "Predict Leaf":
    if os.path.exists(MODEL_SAVE_PATH):
        model = load_trained_model()
        uploaded_file = st.file_uploader("Upload leaf image for prediction", type=['png','jpg','jpeg'])
        if uploaded_file:
            img = Image.open(uploaded_file)
            img_resized = img.resize((128,128))
            X = np.array(img_resized, dtype="float32") / 255.0
            X = X.reshape((1,128,128,3))
            pred = model.predict(X)
            label = "Healthy" if pred[0][0] > pred[0][1] else "Blight"
            st.image(img, caption="Uploaded Leaf", use_column_width=True)
            st.write(f"Prediction: **{label}**")
    else:
        st.warning("No trained model found. Please train a model first.")