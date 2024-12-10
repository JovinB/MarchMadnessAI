from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



def very_simple_MLP(num_features):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(num_features, )), # a fully connected layer with 32 nodes
        Dense(1, activation='sigmoid') # output layer that gives a value from 0 to 1, representing the probability of label 1
    ])

    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    
    return model

def simple_MLP(num_features):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(num_features, )), # a fully connected layer with 64 nodes
        Dense(32, activation='relu'), # a fully connected layer with 32 nodes
        Dense(1, activation='sigmoid') # output layer that gives a value from 0 to 1, representing the probability of label 1
    ])

    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    
    return model

def simple_MLP2(num_features):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(num_features, )), # a fully connected layer with 64 nodes
        Dense(32, activation='relu'), # a fully connected layer with 32 nodes
        Dense(32, activation='relu'), # a fully connected layer with 32 nodes
        Dense(1, activation='sigmoid') # output layer that gives a value from 0 to 1, representing the probability of label 1
    ])

    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    
    return model