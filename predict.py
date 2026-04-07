import joblib
import numpy as np

# Load trained model
model = joblib.load("model_rain.pkl")

def predict_rain(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return prediction, probability


# Manual testing
if __name__ == "__main__":
    sample = [10, 20, 60, 65, 1012, 1010, 15]  # Example values
    
    pred, prob = predict_rain(sample)
    
    print("Rain Prediction:", "Yes" if pred == 1 else "No")
    print("Probability:", round(prob * 100, 2), "%")