import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, 1:]  # Features (256-dimensional)
    y = data[:, 0]   # Labels (digits 0-9)
    return X, y

# Preprocess the data
def preprocess_data(X, y):
    # Normalize features to the range [0, 1]
    X = (X + 1) / 2.0
    # Reshape X to (num_samples, 16, 16, 1) for convolutional layers
    X = X.reshape(-1, 16, 16, 1)
    # Convert labels to one-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=10)
    return X, y

# Fully Connected Neural Network (FCNN)
def build_fcnn():
    model = models.Sequential([
        layers.Input(shape=(256,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Add dropout for regularization
        layers.Dense(64, activation='sigmoid'),
        layers.Dropout(0.5),  # Add dropout for regularization
        layers.Dense(10, activation='softmax')
    ])
    return model

# Convolutional Neural Network (CNN)
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(16, 16, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='tanh', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Add dropout for regularization
        layers.Dense(10, activation='softmax')
    ])
    return model

# Compile and train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=1000, batch_size=32):
    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=10,         # Stop after 10 epochs without improvement
        restore_best_weights=True  # Restore the best model weights
    )

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping])  # Add early stopping
    return history

# Main function
def main():
    # Load and preprocess the data
    X_train, y_train = load_data('zip_train.txt')
    X_test, y_test = load_data('zip_test.txt')
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Build and train the Fully Connected Neural Network (FCNN)
    print("Training Fully Connected Neural Network (FCNN)...")
    fcnn_model = build_fcnn()
    fcnn_history = train_model(fcnn_model, X_train.reshape(-1, 256), y_train, X_val.reshape(-1, 256), y_val)

    # Evaluate FCNN on the test set
    print("Evaluating FCNN on test set...")
    fcnn_test_loss, fcnn_test_acc = fcnn_model.evaluate(X_test.reshape(-1, 256), y_test)
    print(f"FCNN Test Accuracy: {fcnn_test_acc:.4f}")

    # Build and train the Convolutional Neural Network (CNN)
    print("Training Convolutional Neural Network (CNN)...")
    cnn_model = build_cnn()
    cnn_history = train_model(cnn_model, X_train, y_train, X_val, y_val)

    # Evaluate CNN on the test set
    print("Evaluating CNN on test set...")
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test, y_test)
    print(f"CNN Test Accuracy: {cnn_test_acc:.4f}")

if __name__ == "__main__":
    main()