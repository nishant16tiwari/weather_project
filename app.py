from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import warnings
import concurrent.futures
warnings.filterwarnings('ignore')

from feature_engineering import fetch_open_meteo, build_features, resolve_city_state

app = Flask(__name__)

# ── Load model once at startup ─────────────────────────────────────────────
_bundle    = joblib.load("model_rain.pkl")
BOOSTER    = _bundle["booster"]
FEATURES   = _bundle["features"]
THRESHOLD  = float(_bundle["threshold"])
LE_CITY    = _bundle["le_city"]
LE_STATE   = _bundle["le_state"]


# ── Helper: run prediction ─────────────────────────────────────────────────
def predict_from_lat_lon(lat: float, lon: float, city_raw: str):
    """Fetch live data, engineer features, predict. Returns dict."""
    # 1. Resolve city / state
    city, state = resolve_city_state(city_raw)

    # 2. Fetch hourly history
    df = fetch_open_meteo(lat, lon)

    # 3. Build 100 features
    feat_dict, current_weather = build_features(df, city, state, LE_CITY, LE_STATE)

    # 4. Order exactly as training
    row = np.array([[feat_dict[f] for f in FEATURES]], dtype=np.float32)

    # 5. LightGBM Booster.predict → raw probability (1-D array)
    prob_rain = float(BOOSTER.predict(row)[0])

    # 6. Apply custom threshold
    prediction = "Yes" if prob_rain >= THRESHOLD else "No"

    return {
        "prediction":     prediction,
        "probability":    round(prob_rain * 100, 2),
        "current_weather": current_weather,
        "city":           city,
        "state":          state,
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    city_name   = request.form.get("CityName", "").strip()
    lat_str     = request.form.get("Lat",  "").strip()
    lon_str     = request.form.get("Lon",  "").strip()

    # Validate coordinates
    if not lat_str or not lon_str:
        return render_template("index.html",
                               error="Please allow location access before predicting.")
    try:
        lat = float(lat_str)
        lon = float(lon_str)
    except ValueError:
        return render_template("index.html",
                               error="Invalid coordinates. Please try again.")

    try:
        result = predict_from_lat_lon(lat, lon, city_name or "Unknown")
    except Exception as e:
        return render_template("index.html",
                               error=f"Weather API error: {str(e)}")

    cw = result["current_weather"]

    return render_template(
        "result.html",
        prediction   = result["prediction"],
        probability  = result["probability"],
        data         = cw,
        city_name    = city_name or result["city"],
        current_temp = cw["temperature_C"],
        lat          = lat,
        lon          = lon,
    )


@app.route("/extended")
def extended():
    lat       = request.args.get("lat", "")
    lon       = request.args.get("lon", "")
    city_name = request.args.get("city", "")
    return render_template("extended.html", lat=lat, lon=lon, city_name=city_name)


@app.route("/aqi")
def aqi():
    lat       = request.args.get("lat", "")
    lon       = request.args.get("lon", "")
    city_name = request.args.get("city", "")
    return render_template("aqi.html", lat=lat, lon=lon, city_name=city_name)


@app.route("/insights")
def insights():
    lat       = request.args.get("lat", "")
    lon       = request.args.get("lon", "")
    city_name = request.args.get("city", "")
    return render_template("insights.html", lat=lat, lon=lon, city_name=city_name)


GLOBAL_CITIES = [
    {"name": "New York", "lat": 40.7128, "lon": -74.0060, "country": "USA"},
    {"name": "London", "lat": 51.5074, "lon": -0.1278, "country": "UK"},
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "country": "Japan"},
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093, "country": "Australia"},
    {"name": "Rio de Janeiro", "lat": -22.9068, "lon": -43.1729, "country": "Brazil"},
    {"name": "Cape Town", "lat": -33.9249, "lon": 18.4241, "country": "South Africa"},
    {"name": "Dubai", "lat": 25.2048, "lon": 55.2708, "country": "UAE"},
    {"name": "Moscow", "lat": 55.7558, "lon": 37.6173, "country": "Russia"},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "country": "India"},
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522, "country": "France"}
]

@app.route("/global")
def global_radar():
    return render_template("global.html")

@app.route("/api/global_radar")
def api_global_radar():
    results = []
    def fetch_city(city):
        try:
            res = predict_from_lat_lon(city["lat"], city["lon"], city["name"])
            return {
                "name": city["name"],
                "country": city["country"],
                "lat": city["lat"],
                "lon": city["lon"],
                "prediction": res["prediction"],
                "probability": res["probability"],
                "temp": res["current_weather"]["temperature_C"]
            }
        except Exception as e:
            print(f"Global Radar Error for {city['name']}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_city, c) for c in GLOBAL_CITIES]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                results.append(res)
                
    # Sort them back by initial order to be nice, but since it doesn't matter we can just return
    return jsonify({"cities": results})

@app.route("/api/predict_single")
def api_predict_single():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    city = request.args.get("city", "Unknown")
    if not lat or not lon:
        return jsonify({"error": "Missing coordinates"}), 400
    try:
        res = predict_from_lat_lon(float(lat), float(lon), city)
        return jsonify({
            "name": city,
            "lat": float(lat),
            "lon": float(lon),
            "prediction": res["prediction"],
            "probability": res["probability"],
            "temp": res["current_weather"]["temperature_C"],
            "humidity": res["current_weather"]["humidity_pct"],
            "wind": res["current_weather"]["wind_speed_ms"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)