from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
if __name__ == "__main__":
    df = pd.read_csv("../data/sequence_number.txt",header=None,names=['num'],dtype='str')
    sequences = [list(map(int, list(sequence))) for sequence in df["num"]]
    timestep = 5
    X = []
    Y = []
    for i in range(len(sequences) - timestep):
        X.append(sequences[i:timestep + i])
        Y.append(sequences[timestep + i])
    train_input = np.array(X[:140])
    train_targets = np.array(Y[:140])
    test_input = np.array(X[140:])
    test_targets = np.array(X[140:])
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(timestep, 5)))
    model.add(Dense(5))
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(train_input, train_targets, epochs=1000, validation_split=0.1, verbose=1)

