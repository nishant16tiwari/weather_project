"""
feature_engineering.py
Fetches past 48 h of hourly data from Open-Meteo (free, no key needed)
and engineers the 100 features expected by model_rain.pkl.
"""

import math
import json
import urllib.request
import pandas as pd

# ── City → State mapping for known training cities ─────────────────────────
CITY_STATE_MAP = {
    "adilabad":  ("Adilabad",  "Telangana"),
    "agartala":  ("Agartala",  "Tripura"),
    "agra":      ("Agra",      "Uttar_Pradesh"),
    "ahmedabad": ("Ahmedabad", "Gujarat"),
}

def resolve_city_state(city_raw: str):
    """Return (canonical_city, canonical_state) for a raw city string."""
    key = city_raw.strip().lower()
    # Exact match
    if key in CITY_STATE_MAP:
        return CITY_STATE_MAP[key]
    # Partial match
    for k, v in CITY_STATE_MAP.items():
        if k in key or key in k:
            return v
    # Unknown — fallback encoding will be 0
    return (city_raw.strip().title(), "Unknown")


def fetch_open_meteo(lat: float, lon: float) -> pd.DataFrame:
    """Fetch past 2 days + today of hourly weather data from Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
        f"pressure_msl,cloud_cover,wind_speed_10m,wind_direction_10m,"
        f"shortwave_radiation,precipitation,cape,et0_fao_evapotranspiration"
        f"&past_days=2&forecast_days=0&timezone=auto"
    )
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read())

    h = data["hourly"]
    df = pd.DataFrame({
        "time":               pd.to_datetime(h["time"]),
        "temperature_C":      h["temperature_2m"],
        "humidity_pct":       h["relative_humidity_2m"],
        "dew_point_C":        h["dew_point_2m"],
        "pressure_hPa":       h["pressure_msl"],
        "cloud_cover_pct":    h["cloud_cover"],
        # Open-Meteo gives km/h → convert to m/s
        "wind_speed_ms":      [v / 3.6 if v is not None else 0.0
                               for v in h["wind_speed_10m"]],
        "wind_dir":           h["wind_direction_10m"],
        "solar_radiation_Wm2": h["shortwave_radiation"],
        "precip_mm":          h["precipitation"],
        "cape":               h["cape"],
        "et0_mm":             h["et0_fao_evapotranspiration"],
    })

    # Fill any nulls with forward-fill then 0
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_features(df: pd.DataFrame, city: str, state: str,
                   le_city, le_state) -> tuple:
    """
    Engineer all 100 model features from hourly DataFrame.
    Returns (feature_dict, current_weather_dict).
    """
    idx = len(df) - 1   # latest complete hour
    now = df.loc[idx]

    # ── helpers ────────────────────────────────────────────────────────────
    def lag(col, n):
        i = idx - n
        return float(df.loc[max(i, 0), col])

    def roll_mean(col, n):
        start = max(0, idx - n + 1)
        return float(df.loc[start:idx, col].mean())

    def roll_max(col, n):
        start = max(0, idx - n + 1)
        return float(df.loc[start:idx, col].max())

    def roll_std(col, n):
        start = max(0, idx - n + 1)
        vals = df.loc[start:idx, col]
        return float(vals.std()) if len(vals) > 1 else 0.0

    # ── city / state encoding with fallback ────────────────────────────────
    def safe_encode(le, val):
        try:
            return int(le.transform([val])[0])
        except Exception:
            return 0   # unseen label → first encoded index

    city_enc  = safe_encode(le_city,  city)
    state_enc = safe_encode(le_state, state)

    # ── time features ──────────────────────────────────────────────────────
    dt    = now["time"]
    hour  = dt.hour
    month = dt.month
    doy   = dt.timetuple().tm_yday

    T   = float(now["temperature_C"])
    H   = float(now["humidity_pct"])
    P   = float(now["pressure_hPa"])
    D   = float(now["dew_point_C"])
    SR  = float(now["solar_radiation_Wm2"])
    CC  = float(now["cloud_cover_pct"])
    WS  = float(now["wind_speed_ms"])
    WD  = float(now["wind_dir"])
    PP  = float(now["precip_mm"])
    CP  = float(now["cape"])
    ET  = float(now["et0_mm"])

    feat = {
        # ── Base ────────────────────────────────────────────────────────
        "temperature_C":        T,
        "humidity_pct":         H,
        "pressure_hPa":         P,
        "dew_point_C":          D,
        "solar_radiation_Wm2":  SR,
        "cloud_cover_pct":      CC,
        "wind_speed_ms":        WS,
        "wind_dir_sin":         math.sin(math.radians(WD)),
        "wind_dir_cos":         math.cos(math.radians(WD)),
        "cape":                 CP,
        "et0_mm":               ET,
        "city_enc":             city_enc,
        "state_enc":            state_enc,

        # ── Time ────────────────────────────────────────────────────────
        "hour_sin":             math.sin(2 * math.pi * hour  / 24),
        "hour_cos":             math.cos(2 * math.pi * hour  / 24),
        "month_sin":            math.sin(2 * math.pi * month / 12),
        "month_cos":            math.cos(2 * math.pi * month / 12),
        "is_monsoon":           int(6 <= month <= 9),
        "is_night":             int(hour >= 19 or hour < 6),
        "day_of_year":          doy,

        # ── Interactions ────────────────────────────────────────────────
        "humidity_x_dewpoint":  H * D,
        "temp_dewpoint_gap":    T - D,
        "wind_speed_sq":        WS ** 2,
        "cloud_x_solar_inv":    CC * (1.0 - SR / (SR + 1.0)),
        "rel_humidity_approx":  100.0 - 5.0 * (T - D),

        # ── Pressure trends ─────────────────────────────────────────────
        "pressure_trend_1h":    P - lag("pressure_hPa",  1),
        "pressure_trend_3h":    P - lag("pressure_hPa",  3),
        "pressure_trend_6h":    P - lag("pressure_hPa",  6),
        "pressure_trend_24h":   P - lag("pressure_hPa", 24),

        # ── Humidity / cloud trends ─────────────────────────────────────
        "humidity_trend_3h":    H - lag("humidity_pct",    3),
        "humidity_trend_6h":    H - lag("humidity_pct",    6),
        "cloud_trend_3h":       CC - lag("cloud_cover_pct", 3),

        # ── Precip cumulative ───────────────────────────────────────────
        "precip_cumul_6h":      sum(lag("precip_mm", i) for i in range(6)),

        # ── Lags: humidity ──────────────────────────────────────────────
        "humidity_pct_lag1":    lag("humidity_pct",  1),
        "humidity_pct_lag3":    lag("humidity_pct",  3),
        "humidity_pct_lag6":    lag("humidity_pct",  6),
        "humidity_pct_lag12":   lag("humidity_pct", 12),
        "humidity_pct_lag24":   lag("humidity_pct", 24),

        # ── Lags: pressure ──────────────────────────────────────────────
        "pressure_hPa_lag1":    lag("pressure_hPa",  1),
        "pressure_hPa_lag3":    lag("pressure_hPa",  3),
        "pressure_hPa_lag6":    lag("pressure_hPa",  6),
        "pressure_hPa_lag12":   lag("pressure_hPa", 12),
        "pressure_hPa_lag24":   lag("pressure_hPa", 24),

        # ── Lags: cloud ──────────────────────────────────────────────────
        "cloud_cover_pct_lag1":  lag("cloud_cover_pct",  1),
        "cloud_cover_pct_lag3":  lag("cloud_cover_pct",  3),
        "cloud_cover_pct_lag6":  lag("cloud_cover_pct",  6),
        "cloud_cover_pct_lag12": lag("cloud_cover_pct", 12),
        "cloud_cover_pct_lag24": lag("cloud_cover_pct", 24),

        # ── Lags: temperature ────────────────────────────────────────────
        "temperature_C_lag1":   lag("temperature_C",  1),
        "temperature_C_lag3":   lag("temperature_C",  3),
        "temperature_C_lag6":   lag("temperature_C",  6),
        "temperature_C_lag12":  lag("temperature_C", 12),
        "temperature_C_lag24":  lag("temperature_C", 24),

        # ── Lags: dew point ──────────────────────────────────────────────
        "dew_point_C_lag1":     lag("dew_point_C",  1),
        "dew_point_C_lag3":     lag("dew_point_C",  3),
        "dew_point_C_lag6":     lag("dew_point_C",  6),
        "dew_point_C_lag12":    lag("dew_point_C", 12),
        "dew_point_C_lag24":    lag("dew_point_C", 24),

        # ── Lags: precipitation ──────────────────────────────────────────
        "precip_mm_lag1":       lag("precip_mm",  1),
        "precip_mm_lag3":       lag("precip_mm",  3),
        "precip_mm_lag6":       lag("precip_mm",  6),
        "precip_mm_lag12":      lag("precip_mm", 12),
        "precip_mm_lag24":      lag("precip_mm", 24),

        # ── Lags: CAPE ───────────────────────────────────────────────────
        "cape_lag1":            lag("cape",  1),
        "cape_lag3":            lag("cape",  3),
        "cape_lag6":            lag("cape",  6),
        "cape_lag12":           lag("cape", 12),
        "cape_lag24":           lag("cape", 24),

        # ── Lags: solar radiation ────────────────────────────────────────
        "solar_radiation_Wm2_lag1":  lag("solar_radiation_Wm2",  1),
        "solar_radiation_Wm2_lag3":  lag("solar_radiation_Wm2",  3),
        "solar_radiation_Wm2_lag6":  lag("solar_radiation_Wm2",  6),
        "solar_radiation_Wm2_lag12": lag("solar_radiation_Wm2", 12),
        "solar_radiation_Wm2_lag24": lag("solar_radiation_Wm2", 24),

        # ── Rolling: humidity ────────────────────────────────────────────
        "humidity_pct_roll3_mean":  roll_mean("humidity_pct",  3),
        "humidity_pct_roll6_mean":  roll_mean("humidity_pct",  6),
        "humidity_pct_roll12_mean": roll_mean("humidity_pct", 12),
        "humidity_pct_roll3_max":   roll_max ("humidity_pct",  3),
        "humidity_pct_roll6_std":   roll_std ("humidity_pct",  6),

        # ── Rolling: pressure ────────────────────────────────────────────
        "pressure_hPa_roll3_mean":  roll_mean("pressure_hPa",  3),
        "pressure_hPa_roll6_mean":  roll_mean("pressure_hPa",  6),
        "pressure_hPa_roll12_mean": roll_mean("pressure_hPa", 12),
        "pressure_hPa_roll3_max":   roll_max ("pressure_hPa",  3),
        "pressure_hPa_roll6_std":   roll_std ("pressure_hPa",  6),

        # ── Rolling: cloud ───────────────────────────────────────────────
        "cloud_cover_pct_roll3_mean":  roll_mean("cloud_cover_pct",  3),
        "cloud_cover_pct_roll6_mean":  roll_mean("cloud_cover_pct",  6),
        "cloud_cover_pct_roll12_mean": roll_mean("cloud_cover_pct", 12),
        "cloud_cover_pct_roll3_max":   roll_max ("cloud_cover_pct",  3),
        "cloud_cover_pct_roll6_std":   roll_std ("cloud_cover_pct",  6),

        # ── Rolling: precipitation ───────────────────────────────────────
        "precip_mm_roll3_mean":  roll_mean("precip_mm",  3),
        "precip_mm_roll6_mean":  roll_mean("precip_mm",  6),
        "precip_mm_roll12_mean": roll_mean("precip_mm", 12),
        "precip_mm_roll3_max":   roll_max ("precip_mm",  3),
        "precip_mm_roll6_std":   roll_std ("precip_mm",  6),

        # ── Rolling: CAPE ────────────────────────────────────────────────
        "cape_roll3_mean":  roll_mean("cape",  3),
        "cape_roll6_mean":  roll_mean("cape",  6),
        "cape_roll12_mean": roll_mean("cape", 12),
        "cape_roll3_max":   roll_max ("cape",  3),
        "cape_roll6_std":   roll_std ("cape",  6),
    }

    current_weather = {
        "temperature_C":      round(T,  1),
        "humidity_pct":       round(H,  1),
        "pressure_hPa":       round(P,  1),
        "dew_point_C":        round(D,  1),
        "wind_speed_ms":      round(WS, 2),
        "wind_dir":           round(WD, 1),
        "cloud_cover_pct":    round(CC, 1),
        "solar_radiation_Wm2": round(SR, 1),
        "precip_mm":          round(PP, 2),
        "cape":               round(CP, 1),
    }

    return feat, current_weather
