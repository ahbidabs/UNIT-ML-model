import pandas as pd
from train_model import load_models, predict_exercise

def predict_file(file_path):
    # Load the trained models
    clf, le, scaler = load_models()
    
    # Read the test file
    df_test = pd.read_csv(file_path, sep='\t', parse_dates=['time'], engine='python')
    
    # Drop unnecessary columns
    drop_cols = ['DeviceName', 'Version()', 'Temperature(°C)', 'Battery level(%)']
    df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors='ignore')
    
    # Make prediction
    predicted = predict_exercise(clf, le, scaler, df_test)
    print(f"File: {file_path}")
    print(f"Predicted: {predicted}\n")

if __name__ == '__main__':
    # Test on a lat pulldown file
    predict_file('sensormovingshit/sensormovingshit/lat pulldown 3.txt')
    
    # Test on a no exercise file
    predict_file('lat pulldown (medium, 8 reps).txt')
    predict_file('lat pulldown (medium, 8 reps) 2.txt') 
    predict_file('lat pulldown (medium, 8 reps) 1.txt')
    predict_file('Lat pulldown + no exercise/no exercise (lat pulldown, 0 reps) 4.txt')
    predict_file('sensormovingshit/0 reps (swinging).txt')