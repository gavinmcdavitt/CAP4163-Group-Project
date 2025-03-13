import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# A simpler implementation of locally connected layer without weight sharing
class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, kernel_size, stride=1):
        super(LocallyConnected2d, self).__init__()
        input_h, input_w = input_size
        
        # Calculate output dimensions
        output_h = (input_h - kernel_size) // stride + 1
        output_w = (input_w - kernel_size) // stride + 1
        
        # Number of parameters per output location
        self.weight = nn.Parameter(
            torch.randn(out_channels, output_h, output_w, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels, output_h, output_w))
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_size = (output_h, output_w)
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):
        batch_size = x.size(0)
        out_h, out_w = self.output_size
        output = torch.zeros(batch_size, self.out_channels, out_h, out_w, device=x.device)
        
        # Manually process each output location
        for i in range(out_h):
            for j in range(out_w):
                # Extract input patch
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                input_patch = x[:, :, h_start:h_end, w_start:w_end]
                
                # Apply weights at this position to the patch
                for c in range(self.out_channels):
                    # Get the weights for this output channel and position
                    weight = self.weight[c, i, j]
                    # Apply the weights (element-wise multiply and sum)
                    output[:, c, i, j] = (input_patch * weight).sum(dim=(1, 2, 3)) + self.bias[c, i, j]
        
        return output

# Define the locally connected neural network
class LocallyConnectedNet(nn.Module):
    def __init__(self):
        super(LocallyConnectedNet, self).__init__()
        
        # Layer 1: Locally connected layer (16x16 -> 12x12x16)
        self.local1 = LocallyConnected2d(
            in_channels=1, 
            out_channels=16, 
            input_size=(16, 16), 
            kernel_size=5, 
            stride=1
        )
        self.tanh1 = nn.Tanh()
        
        # Layer 2: Locally connected layer (12x12x16 -> 10x10x32)
        self.local2 = LocallyConnected2d(
            in_channels=16, 
            out_channels=32, 
            input_size=(12, 12), 
            kernel_size=3, 
            stride=1
        )
        self.tanh2 = nn.Tanh()
        
        # Layer 3: Locally connected layer (10x10x32 -> 8x8x64)
        self.local3 = LocallyConnected2d(
            in_channels=32, 
            out_channels=64, 
            input_size=(10, 10), 
            kernel_size=3, 
            stride=1
        )
        self.tanh3 = nn.Tanh()
        
        # Layer 4: Fully connected layer (8x8x64 -> 10)
        self.fc = nn.Linear(8 * 8 * 64, 10)
    
    def forward(self, x):
        # Reshape input
        x = x.view(-1, 1, 16, 16)  # Reshape to [batch_size, channels, height, width]
        
        # Apply layers
        x = self.tanh1(self.local1(x))
        x = self.tanh2(self.local2(x))
        x = self.tanh3(self.local3(x))
        
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
    batch_size = 32  # Smaller batch size due to memory constraints
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the network
    model = LocallyConnectedNet().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Using Adam for better convergence
    
    # Define model save path
    model_save_path = 'local_model.pth'
    
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
        plt.savefig('locally_connected_training_loss.png')
        print("Training loss plot saved as 'locally_connected_training_loss.png'")
    
    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Test Accuracy: {accuracy:.4f} ({int(accuracy * y_test.size(0))}/{y_test.size(0)})')
    
    # Plot training loss only if we trained the model
    if not os.path.exists(model_save_path):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, 'b-')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('locally_connected_training_loss.png')
        print("Training loss plot saved as 'locally_connected_training_loss.png'")
    
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
    
    plt.savefig('locally_connected_predictions.png')
    print("Prediction samples saved as 'locally_connected_predictions.png'")
    
if __name__ == "__main__":
    main()