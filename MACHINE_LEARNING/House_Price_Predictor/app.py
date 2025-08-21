from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask import render_template

# Load saved model and encoder
model = joblib.load("models/house_price_model.pkl")
encoder = joblib.load("models/encoder.pkl")

app = Flask(__name__)


def predict_price(title, location, area):
 
    input_df = pd.DataFrame([[title, location]], columns=['title', 'location'])
    input_cat = encoder.transform(input_df)
    input_cat_df = pd.DataFrame(input_cat, columns=encoder.get_feature_names_out(['title','location']), index=[0])
    input_num_df = pd.DataFrame([[area]], columns=['area_insqft'], index=[0])

    input_prepared = pd.concat([input_cat_df, input_num_df], axis=1)

   
    prediction = model.predict(input_prepared)
    return max(0, float(prediction[0]))   


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        title = data.get("title")
        location = data.get("location")
        area = data.get("area")

        if not title or not location or area is None:
            return jsonify({"error": "Please provide title, location, and area"}), 400

        prediction = predict_price(title, location, area)

        return jsonify({
            "title": title,
            "location": location,
            "area_insqft": area,
            "predicted_price(L)": round(prediction, 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
