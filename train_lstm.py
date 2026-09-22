import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

from data_prep import load_and_clean_data, scale_features, create_sequences

def build_model(input_shape):
    model = Sequential([
        LSTM(units=50,input_shape=input_shape,activation='tanh'),
        Dense(units=3)
    ])

    
    model.compile(
        optimizer = 'adam',
        loss='mse'
    )
    return model

if __name__ == "__main__":
    
    df = load_and_clean_data("SmartFarm.csv")
    scaled_array, scaler = scale_features(df, ['soil_moisture', 'temperature', 'humidity'])
    X, y = create_sequences(scaled_array, seq_length=24)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


    shape = (X_train.shape[1], X_train.shape[2]) 
    model = build_model(shape)

   
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size = 32,
        validation_data=(X_test,y_test)
    )

    model.save("farm_lstm.keras")
    print("Model trained and saved to disk!")