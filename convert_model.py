from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np

# Load weights from the old model
old_model = load_model("model.h5", compile=False)
weights = old_model.get_weights()

# Rebuild model structure manually (match your original ANN)
model = Sequential([
    Input(shape=(11,)),
    Dense(6, activation='relu'),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Set weights to the new model
model.set_weights(weights)

# Save in new format
model.save("model_new.keras")
print("✅ Model converted and saved as 'model_new.keras'")
