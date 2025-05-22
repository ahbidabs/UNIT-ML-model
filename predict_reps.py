import os
import pandas as pd
import joblib
from train_rep_model import extract_global_features

def predict_rep_count(file_path):
    """
    Predict the number of reps in an exercise file.
    """
    # Load the trained model
    model = joblib.load("rep_counter.joblib")
    
    # Read the file
    df = pd.read_csv(file_path, sep="\t", parse_dates=['time'])
    df = df.drop(['DeviceName','Version()','Temperature(°C)','Battery level(%)'], axis=1, errors='ignore')
    
    # Extract features
    features = extract_global_features(df)
    features_df = pd.DataFrame([features])
    
    # Make prediction
    predicted_reps = model.predict(features_df)[0]
    
    # Round to nearest whole number since reps are integers
    predicted_reps = round(predicted_reps)
    
    return predicted_reps

if __name__ == "__main__":
    # Test on a few files
    test_files = [
        "rep_data/lat pulldown (3 reps).txt",
        "rep_data/lat pulldown (5 reps) 1.txt",
        "lat pulldown (medium, 8 reps) 1.txt"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            predicted = predict_rep_count(file_path)
            print(f"\nFile: {file_path}")
            print(f"Predicted reps: {predicted}")
        else:
            print(f"\n⚠️ File not found: {file_path}") 