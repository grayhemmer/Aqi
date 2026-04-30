from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("aqi_model.pkl")
df = pd.read_csv("data.csv")
df.columns = df.columns.str.lower()

if 'date' not in df.columns:
    raise Exception("'date' column missing in CSV")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df = df.drop_duplicates(subset='date', keep='last')
df = df.set_index('date')
df = df.asfreq('D')
df['aqi'] = df['aqi'].ffill().bfill()

last_date = df.index.max()

def get_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy"
    elif aqi <= 200: return "Poor"
    else: return "Very Poor"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or 'date' not in data:
            return jsonify({"error": "No date provided"})

        user_date = pd.to_datetime(data['date'])
        days = (user_date - last_date).days

        if days <= 0:
            return jsonify({"error": "Select future date"})
        if days > 30:
            return jsonify({"error": "Max 30 days allowed"})

        full_forecast = model.forecast(steps=days + 3)
        predicted_value = float(full_forecast.iloc[-1])

        start_idx = max(0, len(full_forecast) - 7)
        graph_values = full_forecast.iloc[start_idx:].tolist()
        
        if len(graph_values) < 7:
            graph_values = full_forecast.tolist()

        return jsonify({
            "date": str(user_date.date()),
            "aqi": round(predicted_value, 2),
            "category": get_category(predicted_value),
            "graph_time": [f"Day {i+1}" for i in range(len(graph_values))],
            "graph_aqi": [round(float(i), 2) for i in graph_values]
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)