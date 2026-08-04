import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Resolve paths relative to the project root (not the script location)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure working directories exist
os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE, 'probs'),  exist_ok=True)

# Load preprocessed tensors (already scaled by scaler_lstm.pkl)
X_train = np.load(os.path.join(BASE, 'tensors/X_lstm_train.npy'))  # shape (N, 15, 85)
X_val   = np.load(os.path.join(BASE, 'tensors/X_lstm_val.npy'))
X_test  = np.load(os.path.join(BASE, 'tensors/X_lstm_test.npy'))
y_train = np.load(os.path.join(BASE, 'tensors/y_train.npy'))
y_val   = np.load(os.path.join(BASE, 'tensors/y_val.npy'))

# Build model – exact architecture from the paper (§5.1)
model = Sequential([
    LSTM(128, input_shape=(15, 85), return_sequences=True),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks – checkpoint best model, early stopping on validation loss
ckpt_path = os.path.join(BASE, 'models/lstm_model.h5')
checkpoint = ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=2
)

# Save probabilities for Layer-2 ensemble (train, val, AND test)
probs_train = model.predict(X_train)  # (N_train, 2)
probs_val   = model.predict(X_val)    # (N_val, 2)
probs_test  = model.predict(X_test)   # (N_test, 2)
np.save(os.path.join(BASE, 'probs/lstm_probs_train.npy'), probs_train)
np.save(os.path.join(BASE, 'probs/lstm_probs_val.npy'),   probs_val)
np.save(os.path.join(BASE, 'probs/lstm_probs_test.npy'),  probs_test)

print('✅ LSTM training complete. Model saved to models/lstm_model.h5')
