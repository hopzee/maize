# model.py
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from PIL import Image

IMG_SIZE = (128, 128)
MODEL_SAVE_PATH = "maize_leaf_model.keras"

def train_model(images, labels, epochs=5):
    # Convert images to numpy arrays
    X = np.array([np.array(img.resize(IMG_SIZE)) for img in images], dtype="float32") / 255.0
    y = np.array(labels)
    y = to_categorical(y, num_classes=2)

    # Build a simple CNN
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, batch_size=4)
    
    # Save in .keras format
    model.save(MODEL_SAVE_PATH)
    return model

def load_trained_model():
    return load_model(MODEL_SAVE_PATH)

def infer_label_from_filename(filename):
    # Automatically assign label based on filename
    fname = filename.lower()
    if "healthy" in fname:
        return 0
    elif "blight" in fname:
        return 1
    else:
        raise ValueError("Filename must contain 'healthy' or 'blight'")