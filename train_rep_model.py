import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import savgol_filter, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_DIR = "rep_data"
MODEL_PATH = "rep_counter.joblib"

def extract_rep_count_from_filename(filename):
    # Match patterns like "(8 reps)", "(light, 8 reps)", "(heavy, 8 reps)", etc.
    match = re.search(r"\((?:[^,]*,)?\s*(\d+)\s*reps?\)", filename.lower())
    return int(match.group(1)) if match else None

def extract_global_features(df):
    df = df.rename(columns={
        'AccX(g)': 'acc_x', 'AccY(g)': 'acc_y', 'AccZ(g)': 'acc_z',
        'AsX(°/s)': 'gyr_x', 'AsY(°/s)': 'gyr_y', 'AsZ(°/s)': 'gyr_z'
    })
    df["acc_y"] = pd.to_numeric(df["acc_y"], errors="coerce").fillna(0)
    smoothed = savgol_filter(df["acc_y"], window_length=21, polyorder=3)
    inverted = -smoothed
    peaks, _ = find_peaks(inverted, prominence=0.05, distance=30)

    return {
        "duration_sec": len(df) / 50,
        "energy": np.sum(smoothed**2),
        "peak_count": len(peaks),
        "signal_range": smoothed.max() - smoothed.min(),
        "std_dev": np.std(smoothed)
    }

# Collect training data
X = []
y = []

for file in os.listdir(DATA_DIR):
    if not file.endswith(".txt"):
        continue

    label = extract_rep_count_from_filename(file)
    if label is None:
        print(f"⚠️ Skipping {file} (no rep count found in filename)")
        continue

    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path, sep="\t", parse_dates=['time'])
    df = df.drop(['DeviceName','Version()','Temperature(°C)','Battery level(%)'], axis=1, errors='ignore')

    feats = extract_global_features(df)
    X.append(feats)
    y.append(label)
    print(f"✅ Processed {file} - {label} reps")

# Convert to DataFrame
X_df = pd.DataFrame(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n✅ Model trained and saved to {MODEL_PATH}")
print(f"Number of samples: {len(X)}")
print(f"Features used: {list(X_df.columns)}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Save the model
joblib.dump(model, MODEL_PATH)
