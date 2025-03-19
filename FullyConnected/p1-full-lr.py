import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse

# Define the fully connected neural network
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
        
        # Initialize with Kaiming normal (good default for ReLU networks)
        self.initialize_weights()
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

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

# Function to train and evaluate the model
def train_and_evaluate(learning_rate=0.01, num_epochs=20, batch_size=64, 
                      force_retrain=False, use_early_stopping=True, momentum=0.9):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading training data...")
    X_train, y_train = load_data('zip_train.txt')
    print("Loading test data...")
    X_test, y_test = load_data('zip_test.txt')
    
    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the network
    model = FullyConnectedNet().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    # Define model save path based on parameters
    model_save_path = f'full_model_lr{learning_rate}.pth'
    
    # Training variables
    best_accuracy = 0.0
    early_stop_counter = 0
    early_stop_patience = 5  # Number of epochs to wait before early stopping
    
    # Check if we already have a trained model and aren't forcing a retrain
    if os.path.exists(model_save_path) and not force_retrain:
        print(f"Loading pre-trained model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
    else:
        # Lists to store metrics
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        
        # Time tracking
        start_time = time.time()
        
        # Training loop
        print(f"Starting training with learning rate: {learning_rate}")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            avg_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            train_losses.append(avg_loss)
            train_accuracies.append(train_accuracy)
            
            # Evaluate on test data
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, test_predicted = torch.max(test_outputs, 1)
                test_accuracy = 100 * (test_predicted == y_test).sum().item() / y_test.size(0)
                test_accuracies.append(test_accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
            
            # Early stopping
            if use_early_stopping:
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    # Save the best model
                    torch.save(model.state_dict(), model_save_path)
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # If we didn't use early stopping or never found a better model, save the final model
        if not use_early_stopping or best_accuracy == 0.0:
            torch.save(model.state_dict(), model_save_path)
        else:
            # Load the best model
            model.load_state_dict(torch.load(model_save_path))
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-')
        plt.title(f'Training Loss (LR={learning_rate})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'g-')
        plt.title(f'Training Accuracy (LR={learning_rate})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, 'r-')
        plt.title(f'Test Accuracy (LR={learning_rate})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'full_model_lr{learning_rate}_curves.png')
        print(f"Training curves saved")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0) * 100
        
        # Calculate per-class accuracy
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
        
        print(f'Overall Test Accuracy: {accuracy:.2f}% ({int(accuracy * y_test.size(0) / 100)}/{y_test.size(0)})')
    
    return accuracy, model

# Main function with parameter testing
def main():
    parser = argparse.ArgumentParser(description='Train and evaluate FullyConnectedNet with different learning rates')
    parser.add_argument('--lr', type=float, default=0.01, 
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, 
                       help='Batch size for training')
    parser.add_argument('--momentum', type=float, default=0.9, 
                       help='Momentum for SGD optimizer')
    parser.add_argument('--force-retrain', action='store_true', 
                       help='Force retraining even if a saved model exists')
    parser.add_argument('--no-early-stopping', action='store_true', 
                       help='Disable early stopping')
    
    args = parser.parse_args()
    
    # Train and evaluate with the specified parameters
    accuracy, _ = train_and_evaluate(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        momentum=args.momentum,
        force_retrain=args.force_retrain,
        use_early_stopping=not args.no_early_stopping
    )
    
    print(f"\nExperiment complete. Final accuracy: {accuracy:.2f}%")

# Function to run a series of learning rate experiments
def run_learning_rate_experiments():
    # Learning rates to test
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    
    results = []
    
    for lr in learning_rates:
        print(f"\n\n{'='*80}")
        print(f"Testing learning rate: {lr}")
        print(f"{'='*80}\n")
        
        accuracy, _ = train_and_evaluate(
            learning_rate=lr,
            force_retrain=True  # Force retrain for experiment
        )
        
        results.append({
            'learning_rate': lr,
            'accuracy': accuracy
        })
    
    # Print summary of results
    print("\n\nSummary of Learning Rate Experiments:")
    print(f"{'Learning Rate':<15} {'Accuracy':>10}")
    print(f"{'-'*15} {'-'*10}")
    for result in results:
        print(f"{result['learning_rate']:<15} {result['accuracy']:>10.2f}%")
    
    # Plot learning rate vs accuracy
    plt.figure(figsize=(10, 6))
    plt.semilogx([r['learning_rate'] for r in results], [r['accuracy'] for r in results], 'bo-')
    plt.title('Learning Rate Effect on Accuracy')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.savefig('learning_rate_impact_full.png')
    print("Learning rate impact plot saved as 'learning_rate_impact_full.png'")

if __name__ == "__main__":
    # Uncomment to run learning rate experiments
    run_learning_rate_experiments()
    
    # Regular execution with command line arguments
    main()