import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # Layer 1: Convolutional layer with 32 filters of size 5x5, stride 1, padding 2
        # Input: 16x16x1, Output: 16x16x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Max pooling layer with 2x2 receptive fields and stride 2
        # Input: 16x16x32, Output: 8x8x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Convolutional layer with 64 filters of size 3x3, stride 1, padding 1
        # Input: 8x8x32, Output: 8x8x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()  # Using sigmoid activation as specified
        
        # Layer 4: Max pooling layer with 2x2 receptive fields and stride 2
        # Input: 8x8x64, Output: 4x4x64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 5: Fully connected layer (4x4x64 = 1024 input features to 10 output classes)
        # Input: 1024, Output: 10
        self.fc = nn.Linear(4 * 4 * 64, 10)
    
    def forward(self, x):
        # Reshape input if necessary
        if x.dim() == 2:
            x = x.view(-1, 1, 16, 16)  # Reshape to [batch_size, channels, height, width]
        
        # Apply layers
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.sigmoid(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten and apply fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

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

# Function to visualize a digit
def visualize_digit(digit_vector, title='Digit'):
    # Convert from [-1, 1] to [0, 255]
    pixel_values = (digit_vector + 1) * 127.5
    # Reshape from 256-element vector to 16x16 image
    image = pixel_values.reshape(16, 16)
    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Main execution
def main():
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
    batch_size = 64
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the network
    model = ConvNet().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Define model save path
    model_save_path = 'cnn_model.pth'
    
    # Check if we already have a trained model
    if os.path.exists(model_save_path):
        print(f"Loading pre-trained model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
    else:
        # Training parameters
        num_epochs = 15
        
        # Lists to store metrics
        train_losses = []
        
        # Training loop
        print("Starting training...")
        for epoch in range(num_epochs):
            model.train()
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
                
                # Print progress
                if (i+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        print("Training complete!")
        
        # Save the model
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, 'b-')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('cnn_training_loss.png')
        print("Training loss plot saved as 'cnn_training_loss.png'")
    
    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Test Accuracy: {accuracy:.4f} ({int(accuracy * y_test.size(0))}/{y_test.size(0)})')
    
    # Visualize some predictions
    num_samples_to_visualize = 5
    plt.figure(figsize=(15, 3))
    for i in range(num_samples_to_visualize):
        idx = np.random.randint(0, len(X_test))
        with torch.no_grad():
            test_image = X_test[idx].unsqueeze(0)
            output = model(test_image)
            pred = output.argmax(dim=1).item()
            actual = y_test[idx].item()
        
        plt.subplot(1, num_samples_to_visualize, i+1)
        plt.imshow(X_test[idx].cpu().numpy().reshape(16, 16), cmap='gray')
        plt.title(f'Pred: {pred}, Act: {actual}')
        plt.axis('off')
    
    plt.savefig('cnn_predictions.png')
    print("Prediction samples saved as 'cnn_predictions.png'")
    
if __name__ == "__main__":
    main()