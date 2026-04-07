import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIG — change CSV path after downloading
# ─────────────────────────────────────────────
DATASET_PATH = "data/india_weather.csv"

# ─────────────────────────────────────────────
#  FEATURES your app uses (must exist in dataset)
# ─────────────────────────────────────────────
REQUIRED_FEATURES = [
    'MinTemp', 'MaxTemp',
    'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm',
    'WindSpeed9am'
]
TARGET = 'RainTomorrow'

# ─────────────────────────────────────────────
#  1. LOAD
# ─────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")
print(f"  Columns: {list(df.columns)}\n")

# ─────────────────────────────────────────────
#  2. COLUMN NAME NORMALISATION
#     The India dataset may use different names.
#     Map them to what the app expects.
# ─────────────────────────────────────────────
RENAME_MAP = {
    # Common alternatives → expected name
    'min_temp':      'MinTemp',
    'max_temp':      'MaxTemp',
    'humidity_9am':  'Humidity9am',
    'humidity_3pm':  'Humidity3pm',
    'pressure_9am':  'Pressure9am',
    'pressure_3pm':  'Pressure3pm',
    'wind_speed_9am':'WindSpeed9am',
    'rain_tomorrow': 'RainTomorrow',
    'rainfall_tomorrow': 'RainTomorrow',
    # Add more mappings here if your CSV uses other names
}
df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns}, inplace=True)

# ─────────────────────────────────────────────
#  3. SMART COLUMN BUILDER
#     If exact columns are missing, try to derive them.
# ─────────────────────────────────────────────

# If Pressure columns missing but we have one pressure column, duplicate it
if 'Pressure9am' not in df.columns and 'Pressure3pm' not in df.columns:
    # Look for any pressure-like column
    p_cols = [c for c in df.columns if 'press' in c.lower() or 'slp' in c.lower()]
    if p_cols:
        df['Pressure9am'] = df[p_cols[0]]
        df['Pressure3pm'] = df[p_cols[0]]
        print(f"  ⚠ Derived Pressure columns from: {p_cols[0]}")

# If humidity columns missing look for any humidity column
if 'Humidity9am' not in df.columns and 'Humidity3pm' not in df.columns:
    h_cols = [c for c in df.columns if 'humid' in c.lower() or 'rh' in c.lower()]
    if h_cols:
        df['Humidity9am'] = df[h_cols[0]]
        df['Humidity3pm'] = df[h_cols[0]]
        print(f"  ⚠ Derived Humidity columns from: {h_cols[0]}")

# ─────────────────────────────────────────────
#  4. TARGET COLUMN — handle Yes/No, 1/0, True/False
# ─────────────────────────────────────────────
if TARGET not in df.columns:
    # Try to find a rain target
    rain_cols = [c for c in df.columns if 'rain' in c.lower() and 'today' not in c.lower()]
    if rain_cols:
        df.rename(columns={rain_cols[0]: TARGET}, inplace=True)
        print(f"  ⚠ Using '{rain_cols[0]}' as target column")
    else:
        raise ValueError(f"Cannot find a target column. Available columns: {list(df.columns)}")

# Normalise target to 1/0
if df[TARGET].dtype == object:
    df[TARGET] = df[TARGET].str.strip().str.lower().map({
        'yes': 1, 'no': 0, 'true': 1, 'false': 0,
        '1': 1, '0': 0, 'rain': 1, 'no rain': 0
    })
else:
    df[TARGET] = df[TARGET].astype(int)

# ─────────────────────────────────────────────
#  5. SELECT COLUMNS & CLEAN
# ─────────────────────────────────────────────
missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
if missing:
    raise ValueError(
        f"\n❌ Missing columns: {missing}\n"
        f"   Available columns: {list(df.columns)}\n"
        f"   Add rename mappings in RENAME_MAP above."
    )

df = df[REQUIRED_FEATURES + [TARGET]].copy()
before = len(df)
df.dropna(inplace=True)
print(f"  Dropped {before - len(df):,} rows with missing values")
print(f"  Final dataset: {len(df):,} rows\n")

# ─────────────────────────────────────────────
#  6. CLASS BALANCE CHECK
# ─────────────────────────────────────────────
rain_pct = df[TARGET].mean() * 100
print(f"  Rain days: {rain_pct:.1f}%  |  No-rain days: {100 - rain_pct:.1f}%")
class_weight = 'balanced' if rain_pct < 30 or rain_pct > 70 else None
if class_weight:
    print("  ⚠ Imbalanced dataset — using class_weight='balanced'")

# ─────────────────────────────────────────────
#  7. SPLIT
# ─────────────────────────────────────────────
X = df[REQUIRED_FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────
#  8. TRAIN — tuned RandomForest
# ─────────────────────────────────────────────
print("\nTraining model (this may take 1–2 minutes)...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight=class_weight,
    random_state=42,
    n_jobs=-1       # use all CPU cores
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
#  9. EVALUATE
# ─────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)

print(f"\n{'='*45}")
print(f"  Accuracy : {acc * 100:.2f}%")
print(f"  ROC-AUC  : {auc:.4f}")
print(f"{'='*45}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Rain', 'Rain']))

# ─────────────────────────────────────────────
#  10. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\nFeature Importances:")
importances = zip(REQUIRED_FEATURES, model.feature_importances_)
for feat, imp in sorted(importances, key=lambda x: -x[1]):
    bar = '█' * int(imp * 50)
    print(f"  {feat:<15} {bar} {imp:.4f}")

# ─────────────────────────────────────────────
#  11. SAVE
# ─────────────────────────────────────────────
joblib.dump(model, "model_rain.pkl")
print("\n✅ Model saved as model_rain.pkl")
print("   Restart app.py to use the new model.")