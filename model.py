import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_PATH = "maize_model.pkl"
IMG_SIZE = (64, 64)


def extract_features(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img)

    # flatten image
    return img.flatten()


def preprocess(files):
    X = []
    y = []

    for file in files:
        try:
            img = Image.open(file).convert("RGB")

            features = extract_features(img)
            X.append(features)

            name = file.name.lower()

            if "healthy" in name:
                y.append(0)
            elif "blight" in name:
                y.append(1)

        except:
            continue

    return np.array(X), np.array(y)


def train_model(files):
    X, y = preprocess(files)

    if len(X) == 0:
        raise ValueError("No valid images.")

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    return model


def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, file):
    img = Image.open(file).convert("RGB")

    features = extract_features(img).reshape(1, -1)

    pred = model.predict(features)[0]

    if pred == 1:
        return "Blight"
    else:
        return "Healthy"
