import numpy as np
import data_config
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Masking, RNN
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# Custom GRU Cell using given equations:
# z_t = σ(W_z [x_t, h_(t-1)])
# r_t = σ(W_r [x_t, h_(t-1)])
# ĥ_t = tanh(W [x_t, (r_t * h_(t-1))])
# h_t = z_t * h_(t-1) + (1 - z_t) * ĥ_t
# -------------------------------------------------------------
class MyGRUCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(MyGRUCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = self.units  # Required by Keras to know the state shape
        self.output_size = self.units # Typically equal to state_size for GRU

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.Wz = self.add_weight(shape=(input_dim + self.units, self.units),
                                  initializer='glorot_uniform', name='Wz')
        self.bz = self.add_weight(shape=(self.units,),
                                  initializer='zeros', name='bz')

        self.Wr = self.add_weight(shape=(input_dim + self.units, self.units),
                                  initializer='glorot_uniform', name='Wr')
        self.br = self.add_weight(shape=(self.units,),
                                  initializer='zeros', name='br')

        self.Wh = self.add_weight(shape=(input_dim + self.units, self.units),
                                  initializer='glorot_uniform', name='Wh')
        self.bh = self.add_weight(shape=(self.units,),
                                  initializer='zeros', name='bh')

        self.built = True

    def call(self, inputs, states):
        h_prev = states[0]  # Previous hidden state

        # Concatenate x_t and h_(t-1)
        combined = tf.concat([inputs, h_prev], axis=1)

        # Update gate
        z_t = tf.nn.sigmoid(tf.matmul(combined, self.Wz) + self.bz)

        # Reset gate
        r_t = tf.nn.sigmoid(tf.matmul(combined, self.Wr) + self.br)

        # Candidate hidden state
        combined_reset = tf.concat([inputs, r_t * h_prev], axis=1)
        h_hat_t = tf.nn.tanh(tf.matmul(combined_reset, self.Wh) + self.bh)

        # New hidden state
        h_t = z_t * h_prev + (1 - z_t) * h_hat_t

        return h_t, [h_t]

    def get_config(self):
        config = super(MyGRUCell, self).get_config()
        config.update({"units": self.units})
        return config

# -------------------------------------------------------------
# Load and prepare the data
# -------------------------------------------------------------
train_data = data_config.get_data(2003, 2021)  # Example range: adjust as needed
test_data = data_config.get_data(2022, 2023)

# The get_data output is assumed to be: [ ..., label ]
# Where label is the last column.
X_train = train_data[:, :-1]
y_train = train_data[:, -1]
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
# Build the model using our custom GRU cell
# -------------------------------------------------------------
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(sequence_length, num_features)))
model.add(RNN(MyGRUCell(64), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_seq, y_train_seq, 
          epochs=1000, 
          batch_size=32, 
          validation_split=0.1)

# Evaluate on test set
loss, accuracy = model.evaluate(X_test_seq, y_test_seq)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Predict on 2022-2023 season matches
y_pred = model.predict(X_test_seq)
y_pred_classes = (y_pred > 0.5).astype(int)

# Now y_pred_classes contains the predictions for the sequences from 2022-2023 season