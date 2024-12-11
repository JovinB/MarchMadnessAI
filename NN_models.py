from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GRU

def simple_MLP(num_features):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(num_features, )),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid') # output layer that gives a value from 0 to 1, representing the probability of label 1
    ])

    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    
    return model

def longer_MLP(num_features):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(num_features, )),
        Dense(10, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    
    return model

def longest_MLP(num_features):
    model = Sequential([
        Dense(200, activation='relu', input_shape=(num_features, )),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    
    return model

def longest_MLP_dropout(num_features):
    model = Sequential([
        Dense(200, activation='relu', input_shape=(num_features, )),
        Dense(100, activation='relu'),
        Dropout(0.1),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dropout(0.1),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    
    return model

def longest_MLP_BN(num_features):
    model = Sequential([
        Dense(200, activation='relu', input_shape=(num_features, )),
        BatchNormalization(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        BatchNormalization(),
        Dense(25, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    
    return model

def GRU(num_features):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(num_features, )),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    
    return model


