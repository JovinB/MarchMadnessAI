import numpy as np
import data_config
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense, Masking
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# Assuming the functions and code you provided are available:
# - get_data(start_year, end_year)
# - create_team_sequences(data, sequence_length)
# - etc.
# -------------------------------------------------------------

# Step 1: Load the data
# We will use all data up to 2021 as training, and predict on 2022-2023
train_data = data_config.get_data(2003, 2021)  # Example range: adjust as needed
test_data = data_config.get_data(2022, 2023)

# The get_data output is assumed to be: [ ..., label ]
# Where label is at the end. For demonstration, we assume:
# game_data = [seed_winner_or_loser, team_stats..., seed_opponent, opponent_stats..., label]
# If your data is structured differently, adjust indexing accordingly.

# Let's separate features and labels
X_train = train_data[:, :-1]  # all but last column are features
y_train = train_data[:, -1]   # last column is label
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# -------------------------------------------------------------
# Creating sequences
#
# If you want to use `create_team_sequences`, you need the data 
# in a format like:
# [
#    [winner_team_id, loser_team_id, feature_1, feature_2, ...],
#    ...
# ]
#
# If the data is not in this format, you need to adapt it.
# Below is a conceptual example. Adjust as necessary.

# Suppose you originally have access to (winner_team_id, loser_team_id)
# For demonstration, let's say you have them. If not, modify the pipeline to 
# keep track of teams in the data array.

# Example: If your pipeline doesn't produce that format, consider that 
# you may not need `create_team_sequences`. 
# Instead, you could just reshape your data into sequences if 
# you have time-dependent data.

sequence_length = 10  # Example sequence length
# Example assumption: Each row in X_train corresponds to a single game. 
# To use sequences, you'd need to group games by team and order them in time.
# This is non-trivial because you must identify which rows correspond 
# to which team and then create a consistent sequence.

# For now, let's assume you have a function or a way to produce sequences of shape:
# (num_samples, sequence_length, num_features)
#
# For simplicity, let's skip the team-based sequence creation and 
# just simulate creating sequences from consecutive games. In a real scenario,
# you'd want to group by team and sort by date, then create sequences.

def create_sequences(X, y, seq_length=10):
    X_seqs = []
    y_seqs = []
    for i in range(len(X)-seq_length+1):
        X_seqs.append(X[i:i+seq_length])
        # Typically, if predicting a final outcome, you might use the label 
        # after the sequence, but let's just align them for simplicity:
        y_seqs.append(y[i+seq_length-1])
    return np.array(X_seqs), np.array(y_seqs)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

# -------------------------------------------------------------
# Build the GRU model
#
# Dimensions:
# X_train_seq.shape = (num_samples, sequence_length, num_features)
num_features = X_train_seq.shape[2]

model = Sequential()
# If sequences contain zeros for padding (optional), a Masking layer can help the GRU ignore them
model.add(Masking(mask_value=0.0, input_shape=(sequence_length, num_features)))
model.add(GRU(64, return_sequences=False))  # 64 units as an example
model.add(Dense(1, activation='sigmoid'))   # Binary classification

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
