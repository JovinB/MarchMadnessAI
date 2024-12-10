import numpy as np
import data_config
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Masking, RNN
from sklearn.model_selection import train_test_split

# --------------------------------------------
# Define a custom GRU Cell in TensorFlow Keras
# --------------------------------------------
class MyGRUCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(MyGRUCell, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # input_shape: (batch, time, features)
        input_dim = input_shape[-1] + self.units  # because we concat x_t and h_{t-1}
        
        # Weights for update gate z_t
        self.W_z = self.add_weight(shape=(input_dim, self.units),
                                   initializer='glorot_uniform',
                                   name='W_z')
        self.b_z = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   name='b_z')

        # Weights for reset gate r_t
        self.W_r = self.add_weight(shape=(input_dim, self.units),
                                   initializer='glorot_uniform',
                                   name='W_r')
        self.b_r = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   name='b_r')

        # Weights for candidate hidden state ĥ_t
        self.W_h = self.add_weight(shape=(input_dim, self.units),
                                   initializer='glorot_uniform',
                                   name='W_h')
        self.b_h = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   name='b_h')

        self.built = True

    def call(self, inputs, states):
        # inputs: x_t of shape (batch, features)
        # states: [h_{t-1}] of shape (batch, units)
        
        h_prev = states[0]
        
        # Concatenate input and previous hidden state
        concat_x_h = tf.concat([inputs, h_prev], axis=-1)

        # Update gate
        z_t = tf.nn.sigmoid(tf.matmul(concat_x_h, self.W_z) + self.b_z)

        # Reset gate
        r_t = tf.nn.sigmoid(tf.matmul(concat_x_h, self.W_r) + self.b_r)
        
        # Candidate hidden state
        # For the candidate, we apply reset gate to h_prev before concatenation
        concat_x_rh = tf.concat([inputs, r_t * h_prev], axis=-1)
        h_hat_t = tf.nn.tanh(tf.matmul(concat_x_rh, self.W_h) + self.b_h)

        # Final hidden state
        h_t = z_t * h_prev + (1 - z_t) * h_hat_t

        return h_t, [h_t]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # Return initial state of zeros
        return [tf.zeros((batch_size, self.units), dtype=dtype)]


# -------------------------------------------------------------
# Using the provided code to get and prepare data
# -------------------------------------------------------------
# Step 1: Load the data
# We will use all data up to 2021 as training, and predict on 2022-2023
train_data = data_config.get_data(2003, 2021)  # Example range: adjust as needed
test_data = data_config.get_data(2022, 2023)

# The get_data output is assumed to be: [ ..., label ]
X_train = train_data[:, :-1]  # all but last column are features
y_train = train_data[:, -1]   # last column is label
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

sequence_length = 10  # Example sequence length

def create_sequences(X, y, seq_length=10):
    X_seqs = []
    y_seqs = []
    for i in range(len(X)-seq_length+1):
        X_seqs.append(X[i:i+seq_length])
        y_seqs.append(y[i+seq_length-1])
    return np.array(X_seqs), np.array(y_seqs)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

num_features = X_train_seq.shape[2]

# -------------------------------------------------------------
# Build and train the model using the custom GRU cell
# -------------------------------------------------------------
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(sequence_length, num_features)))
# Instead of model.add(GRU(64)), we use our custom GRU cell wrapped in an RNN layer
model.add(RNN(MyGRUCell(64), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_seq, y_train_seq, 
          epochs=10,    # Adjust epochs as needed (set to 10 for quick demo)
          batch_size=32, 
          validation_split=0.1)

# Evaluate on test set
loss, accuracy = model.evaluate(X_test_seq, y_test_seq)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Predict on 2022-2023 season matches
y_pred = model.predict(X_test_seq)
y_pred_classes = (y_pred > 0.5).astype(int)

print("Predictions for 2022-2023 season (class labels):")
print(y_pred_classes)
