import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Keywords indicating no exercise sessions
NO_EXERCISE_KEYWORDS = ['no exercise', '0 reps', 'setup', 'failed', 'random']

# Only one exercise: lat pulldown
EXERCISE_KEYWORDS = {
    'lat_pulldown': ['lat pulldown']
}

# File paths for saving models
MODEL_DIR = 'models'
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'rf_classifier.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)

def save_models(clf, le, scaler):
    ensure_model_dir()
    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(le, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved classifier to {CLASSIFIER_PATH}")
    print(f"Saved encoder to {ENCODER_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")

def load_models():
    clf = joblib.load(CLASSIFIER_PATH)
    le = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, le, scaler

def load_and_label_data(data_dir, file_pattern="*.txt"):
    """
    Load txt files, label as 'lat_pulldown' or 'no_exercise'
    based on filename keywords.
    """
    all_files = glob.glob(os.path.join(data_dir, file_pattern))
    data_list, labels = [], []
    for f in all_files:
        name = os.path.basename(f).lower()
        # Check no exercise first
        if any(kw in name for kw in NO_EXERCISE_KEYWORDS):
            lbl = 'no_exercise'
        # Then lat pulldown
        elif any(kw in name for kw in EXERCISE_KEYWORDS['lat_pulldown']):
            lbl = 'lat_pulldown'
        else:
            # Skip any other files
            continue

        # Read file
        df = pd.read_csv(f, sep='\t', parse_dates=['time'], engine='python')
        # Drop unwanted columns
        drop_cols = ['time', 'DeviceName', 'Version()', 'Temperature(°C)', 'Battery level(%)']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        data_list.append(df)
        labels.append(lbl)

    return data_list, labels

def extract_features(df, window_size=25, step_size=10):
    """
    Sliding-window feature extraction: mean, std, min, max, range, zero-crossings, acc_r, gyr_r.
    """
    df = df.copy()

    # Add magnitude features
    if all(col in df.columns for col in ['AccX(g)', 'AccY(g)', 'AccZ(g)']):
        df['acc_r'] = np.sqrt(df['AccX(g)']**2 + df['AccY(g)']**2 + df['AccZ(g)']**2)
    if all(col in df.columns for col in ['AsX(°/s)', 'AsY(°/s)', 'AsZ(°/s)']):
        df['gyr_r'] = np.sqrt(df['AsX(°/s)']**2 + df['AsY(°/s)']**2 + df['AsZ(°/s)']**2)

    num_df = df.select_dtypes(include=[np.number])
    windows = []
    for start in range(0, len(num_df) - window_size + 1, step_size):
        w = num_df.iloc[start:start + window_size]
        feat = {}
        for col in num_df.columns:
            series = w[col]
            feat[f"{col}_mean"] = series.mean()
            feat[f"{col}_std"] = series.std()
            feat[f"{col}_min"] = series.min()
            feat[f"{col}_max"] = series.max()
            feat[f"{col}_range"] = series.max() - series.min()
            # Use 1.0-crossings for acc_y, 0.0-crossings otherwise
            baseline = 1.0 if col == "acc_y" else 0.0
            crossings = ((series - baseline) * (series.shift(1) - baseline) < 0).sum()
            feat[f"{col}_crossings"] = crossings

        windows.append(feat)
    return pd.DataFrame(windows)


def build_dataset(data_list, labels, window_size=25, step_size=10):
    X_parts, y_parts = [], []
    for df, lbl in zip(data_list, labels):
        feats = extract_features(df, window_size, step_size)
        if feats.empty:
            continue
        X_parts.append(feats)
        y_parts.extend([lbl] * len(feats))
    X = pd.concat(X_parts, ignore_index=True)
    y = np.array(y_parts)
    return X, y

def train_classifier(X, y, save=True):
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    if save:
        save_models(clf, le, scaler)
    return clf, le, scaler

def predict_exercise(clf, le, scaler, df, window_size=25, step_size=10):
    # Prepare df
    df_proc = df.copy()
    if 'time' in df_proc.columns:
        df_proc = df_proc.drop('time', axis=1)
    # Feature extraction
    feats = extract_features(df_proc.select_dtypes(include=[np.number]), window_size, step_size)
    X_scaled = scaler.transform(feats)
    preds = clf.predict(X_scaled)
    # Return most frequent
    common = np.bincount(preds)
    top = common.argmax()
    return le.inverse_transform([top])[0]

if __name__ == '__main__':
    # 1) Train and save model
    data_dir = 'Lat pulldown + no exercise'
    data_list, labels = load_and_label_data(data_dir)
    X, y = build_dataset(data_list, labels)
    train_classifier(X, y)

    # 2) Test on a sample file
    test_file = 'sensormovingshit/sensormovingshit/lat pulldown 5.txt'
    clf, le, scaler = load_models()
    df_test = pd.read_csv(test_file, sep='\t', parse_dates=['time'], engine='python')
    drop_cols = ['DeviceName', 'Version()', 'Temperature(°C)', 'Battery level(%)']
    df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors='ignore')
    predicted = predict_exercise(clf, le, scaler, df_test)
    print(f"Predicted: {predicted}") 