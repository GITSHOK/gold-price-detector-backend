from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

# Load the trained model
model = joblib.load("gold_price_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expecting JSON input: {"features": [x1, x2, x3, x4]}
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        
        prediction = model.predict(features)
        return jsonify({"prediction": float(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
