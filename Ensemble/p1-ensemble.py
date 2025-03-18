import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import copy

# Import our model architectures from the previous scripts
# If you've modified these classes in the original files, make sure to update them here too

# Fully connected network
class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

# Locally connected network (simplified for this demonstration)
class LocallyConnectedNet(nn.Module):
    def __init__(self):
        super(LocallyConnectedNet, self).__init__()
        # We'll use a simplified version here since the original implementation is complex
        # In practice, you would import your actual locally connected implementation
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)  # Not truly locally connected, but a substitute
        self.tanh1 = nn.Tanh()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.tanh2 = nn.Tanh()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.tanh3 = nn.Tanh()
        self.fc = nn.Linear(64 * 4 * 4, 10)  # Adjusted size based on the convolutions
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 16, 16)
        x = self.tanh1(self.conv1(x))
        x = self.tanh2(self.conv2(x))
        x = self.tanh3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# CNN network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(4 * 4 * 64, 10)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 16, 16)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Ensemble class that combines multiple models
class EnsembleModel:
    def __init__(self, models, voting='soft'):
        self.models = models
        self.voting = voting  # 'hard' or 'soft'
        
    def predict(self, x):
        predictions = []
        
        # Get predictions from each model
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                if self.voting == 'soft':
                    # Soft voting: average the probabilities
                    predictions.append(torch.softmax(logits, dim=1))
                else:
                    # Hard voting: predict class directly
                    _, pred = torch.max(logits, 1)
                    predictions.append(pred)
        
        # Combine predictions
        if self.voting == 'soft':
            # Average the probabilities
            ensemble_pred = torch.stack(predictions).mean(dim=0)
            # Get the class with highest probability
            _, final_pred = torch.max(ensemble_pred, 1)
        else:
            # Count votes for each class
            stacked = torch.stack(predictions)
            # For each sample, count occurrences of each class
            final_pred = torch.mode(stacked, dim=0).values
            
        return final_pred

# Function to load data from the specified file
def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            labels.append(int(float(values[0])))  # First value is the label
            features = [float(val) for val in values[1:]]  # Remaining values are features
            data.append(features)
    return np.array(data), np.array(labels)

# Function to load pre-trained models
def load_models(device):
    models = []
    
    # Try to load the fully connected model
    fc_model = FullyConnectedNet().to(device)
    fc_path = 'full_model.pth'
    if os.path.exists(fc_path):
        print(f"Loading fully connected model from {fc_path}")
        fc_model.load_state_dict(torch.load(fc_path))
        models.append(fc_model)
    else:
        print(f"Fully connected model file {fc_path} not found")
    
    # Try to load the locally connected model
    lc_model = LocallyConnectedNet().to(device)
    lc_path = 'local_model.pth'
    if os.path.exists(lc_path):
        print(f"Loading locally connected model from {lc_path}")
        lc_model.load_state_dict(torch.load(lc_path))
        models.append(lc_model)
    else:
        print(f"Locally connected model file {lc_path} not found")
    
    # Try to load the CNN model
    cnn_model = ConvNet().to(device)
    cnn_path = 'cnn_model.pth'
    if os.path.exists(cnn_path):
        print(f"Loading CNN model from {cnn_path}")
        cnn_model.load_state_dict(torch.load(cnn_path))
        models.append(cnn_model)
    else:
        print(f"CNN model file {cnn_path} not found")
    
    # If we don't have any models, train some
    if len(models) == 0:
        print("No pre-trained models found. Training new models...")
        models = train_new_models(device)
    
    return models

# Function to train new models if needed
def train_new_models(device):
    models = []
    
    # Load data
    print("Loading training data...")
    X_train, y_train = load_data('zip_train.txt')
    print("Loading test data...")
    X_test, y_test = load_data('zip_test.txt')
    
    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    
    # Create 3 variations of the fully connected network
    for i in range(3):
        model = FullyConnectedNet().to(device)
        
        # Use different random initializations
        for param in model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.1 * (i + 1))
        
        # Train the model
        print(f"Training model {i+1}/3...")
        train_model(model, X_train, y_train, num_epochs=10, batch_size=64, 
                   learning_rate=0.01, model_save_path=f'ensemble_model_{i+1}.pth')
        
        models.append(model)
    
    return models

# Function to train a single model
def train_model(model, X_train, y_train, num_epochs=10, batch_size=64, learning_rate=0.01, model_save_path=None):
    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Save the model if a path is provided
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    
    return model

# Function to evaluate individual models and ensemble
def evaluate_models(models, ensemble, X_test, y_test):
    results = {}
    
    # Evaluate each individual model
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0) * 100
            results[f'Model {i+1}'] = accuracy
            print(f'Model {i+1} Accuracy: {accuracy:.2f}%')
    
    # Evaluate the ensemble with soft voting
    ensemble.voting = 'soft'
    predicted = ensemble.predict(X_test)
    soft_accuracy = (predicted == y_test).sum().item() / y_test.size(0) * 100
    results['Ensemble (Soft)'] = soft_accuracy
    print(f'Ensemble (Soft Voting) Accuracy: {soft_accuracy:.2f}%')
    
    # Evaluate the ensemble with hard voting
    ensemble.voting = 'hard'
    predicted = ensemble.predict(X_test)
    hard_accuracy = (predicted == y_test).sum().item() / y_test.size(0) * 100
    results['Ensemble (Hard)'] = hard_accuracy
    print(f'Ensemble (Hard Voting) Accuracy: {hard_accuracy:.2f}%')
    
    # Calculate per-class accuracy for the best method (soft or hard voting)
    best_voting = 'soft' if soft_accuracy > hard_accuracy else 'hard'
    ensemble.voting = best_voting
    predicted = ensemble.predict(X_test)
    
    class_correct = [0] * 10
    class_total = [0] * 10
    for i in range(len(y_test)):
        label = y_test[i].item()
        class_total[label] += 1
        if predicted[i].item() == label:
            class_correct[label] += 1
    
    # Print per-class accuracy
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'Accuracy of digit {i}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')
    
    return results

# Plot bar chart comparing model performances
def plot_results(results):
    plt.figure(figsize=(12, 6))
    
    # Extract model names and accuracies
    models = list(results.keys())
    accuracies = list(results.values())
    
    # Create bar chart
    bars = plt.bar(models, accuracies, color=['blue', 'blue', 'blue', 'green', 'red'])
    
    # Color the ensemble bars differently
    if len(bars) >= 4:
        bars[-2].set_color('green')
        bars[-1].set_color('red')
    
    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom', rotation=0)
    
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.ylim(min(accuracies) - 5, max(accuracies) + 5)  # Add some padding
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('ensemble_comparison.png')
    print("Performance comparison plot saved as 'ensemble_comparison.png'")

# Main function
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data for evaluation...")
    _, y_test_np = load_data('zip_test.txt')
    X_test, y_test = load_data('zip_test.txt')
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    
    # Load or train models
    models = load_models(device)
    
    # Create ensemble
    ensemble = EnsembleModel(models)
    
    # Evaluate models and ensemble
    results = evaluate_models(models, ensemble, X_test, y_test)
    
    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()