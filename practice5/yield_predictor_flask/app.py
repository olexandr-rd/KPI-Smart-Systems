from flask import Flask, request, jsonify, render_template
import joblib
import json
import os

app = Flask(__name__)

MODEL_PATH = "./practice5/model.pkl"
DATA_PATH = "./practice5/data.json"

model = joblib.load(MODEL_PATH)

def predict_yield(temp, humidity, soil):
    features = [[temp, humidity, soil]]
    predicted = model.predict(features)[0]
    return round(predicted, 2)

def load_data():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def save_data(new_entry):
    data = load_data()
    data.append(new_entry)
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

@app.route("/", methods=["GET"])
def index():
    data = load_data()
    return render_template("index.html", data=data)

@app.route("/data", methods=["GET"])
def get_data():
    data = load_data()
    return jsonify(data)

@app.route("/submit", methods=["POST"])
def submit():
    content = request.json
    temp = content.get("temp")
    humidity = content.get("humidity")
    soil = content.get("soil")

    if temp is None or humidity is None or soil is None:
        return jsonify({"error": "Missing data"}), 400

    predicted = predict_yield(temp, humidity, soil)
    entry = {
        "temp": temp,
        "humidity": humidity,
        "soil": soil,
        "yield": predicted
    }
    save_data(entry)
    return jsonify(entry), 200

if __name__ == "__main__":
    app.run(debug=True)
