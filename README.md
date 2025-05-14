# House-Price-Prediction-using-ANN

This project leverages Artificial Neural Networks (ANN) to predict house prices based on features like area, location, number of rooms, and more. Built using Python, Pandas, and TensorFlow/Keras, the model is trained on a structured dataset with the goal of minimizing prediction error and improving real-world estimation accuracy.

Key Features:
Data Preprocessing: Cleaned and normalized data for optimal ANN performance.

Model Architecture: Multi-layer perceptron with ReLU activation and dropout layers to prevent overfitting.

Evaluation: Trained and tested with metrics such as Mean Absolute Error (MAE) and R² Score.

Visualization: Includes loss and accuracy plots to track model performance during training.

User-Focused: Designed to be easily extended or deployed for real-estate-based applications.

# code
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.datasets import fetch_california_housing

from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)  # Features
y = housing.target  # Target (house price)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Define the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), input_shape=(X_train.shape[1],)),  # Hidden layer 1 with L2 regularization
    keras.layers.Dropout(0.3),  # Increased dropout to reduce overfitting
    keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),  # Hidden layer 2 with L2 regularization
    keras.layers.Dropout(0.3),  
    keras.layers.Dense(16, activation='relu'),  # Hidden layer 3
    keras.layers.Dense(1)  # Output layer for regression (no activation)
])
# Compile the model with Adam optimizer and reduced learning rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Reduced learning rate
              loss='mse', 
              metrics=['mae'])
              # Add Early Stopping to avoid overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train the model with validation data
history = model.fit(X_train, y_train, epochs=100, batch_size=16, 
                    validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])
model.summary()
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"📉 Test MAE: {test_mae:.2f}")  # Lower MAE is better
# Predict house prices
y_pred = model.predict(X_test)

# Show first 5 predictions
print("Predicted Prices:", np.round(y_pred[:5].flatten(), 2))
print("Actual Prices:", y_test[:5])
# Predict house prices using the trained model
y_pred = model.predict(X_test)

# Convert predictions to a readable format (flatten to 1D array)
y_pred = y_pred.flatten()

# Display the first 5 actual vs. predicted values
print("📌 First 5 House Price Predictions:")
for i in range(5):
    print(f"🏠 House {i+1}: Predicted Price = ${y_pred[i]:,.2f}, Actual Price = ${y_test[i]:,.2f}")

# If using scaled data, remember to inverse transform the predictions!
# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("Training & Validation Loss")
plt.show()

# Plot MAE
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.title("Training & Validation MAE")
plt.show()
