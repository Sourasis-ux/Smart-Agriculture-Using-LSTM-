import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(csv_path):
    df=pd.read_csv(csv_path)
    df=df.ffill()
    return df

def scale_features(df, columns_to_scale):
    minmax = MinMaxScaler()
    scaled_data = minmax.fit_transform(df[columns_to_scale])
    return scaled_data,minmax

def create_sequences(scaled_data, seq_length=24):
    X = []
    y = []
    
    for i in range(len(scaled_data)-seq_length):
        start=i
        end = i+seq_length

        X.append(scaled_data[start:end])
        y.append(scaled_data[end])
    

    return np.array(X), np.array(y)

if __name__ == "__main__":
   
    df = load_and_clean_data("SmartFarm.csv")
    
   
    features = ['soil_moisture', 'temperature', 'humidity']
    scaled_array, scaler = scale_features(df, features)
    
    
    X, y = create_sequences(scaled_array, seq_length=24)
    
    print("Data loaded and cleaned successfully!")
    print("X shape:", X.shape) # Expected: (samples, 24, 3)
    print("y shape:", y.shape) # Expected: (samples, 3)
