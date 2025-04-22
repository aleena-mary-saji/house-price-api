from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("house_price_model.pkl")

@app.route('/')
def home():
    return "🏠 House Price Predictor API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    rm = data.get("rm")
    lstat = data.get("lstat")
    ptratio = data.get("ptratio")

    prediction = model.predict([[rm, lstat, ptratio]])
    return jsonify({"predicted_price": round(prediction[0], 2)})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

