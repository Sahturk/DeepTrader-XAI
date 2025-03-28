from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import pandas as pd

class PricePredictor:
    def __init__(self):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, train_data):
        self.model.fit(train_data, epochs=5, batch_size=1, verbose=2)

    def predict(self, data):
        return self.model.predict(data)