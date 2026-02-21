import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model

# Load models
disease_model = load_model("../models/disease_model.h5")
class_names = joblib.load("../models/disease_classes.pkl")

yield_model = joblib.load("../models/yield_model.pkl")
yield_features = joblib.load("../models/yield_features.pkl")


# ---------- Disease Prediction ----------
def predict_disease(image):
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 3)

    pred = disease_model.predict(img)
    class_idx = np.argmax(pred)
    return class_names[class_idx]


# ---------- Yield Prediction ----------
def predict_yield(input_dict):
    import pandas as pd
    
    df = pd.DataFrame([input_dict])
    
    # Align columns with training
    for col in yield_features:
        if col not in df:
            df[col] = 0

    df = df[yield_features]
    prediction = yield_model.predict(df)[0]
    return round(prediction, 2)
