import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
 
# Step 1: Create a simple dataset for demonstration (simple X values and random Y values)
torch.manual_seed(42)  # For reproducibility
X = torch.linspace(-1, 1, 100).reshape(-1, 1)  # 100 data points between -1 and 1
Y = torch.rand(100, 1)  # Random target values
 
# Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define the architecture (input to hidden layers and output layer)
        self.hidden1 = nn.Linear(1, 10)  # Input layer (1 feature) to first hidden layer (10 neurons)
        self.hidden2 = nn.Linear(10, 10)  # First hidden to second hidden layer (10 neurons)
        self.hidden3 = nn.Linear(10, 5)   # Second hidden to third hidden layer (5 neurons)
        self.output = nn.Linear(5, 1)     # Third hidden to output layer (1 output)
 
        # Initialize weights and biases with random values using torch.rand
        for layer in [self.hidden1, self.hidden2, self.hidden3, self.output]:
            layer.weight.data = torch.rand(layer.weight.size())
            layer.bias.data = torch.rand(layer.bias.size())
 
    def forward(self, x):
        # Define the forward pass with Sigmoid for hidden layers and Tanh for output
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = torch.sigmoid(self.hidden3(x))
        x = torch.tanh(self.output(x))
        return x
 
# Instantiate the network
net = SimpleNN()
 
# Make initial predictions before training
with torch.no_grad():  # No need to calculate gradients for predictions
    initial_predictions = net(X)
 
# Visualization of the initial predictions
sns.lineplot(x=X.squeeze().numpy(), y=initial_predictions.squeeze().numpy())
plt.title('Initial Predictions (Before Training)')
plt.xlabel('Input X')
plt.ylabel('Predicted Y')
plt.show()
 
# Step 2: Make the Network Trainable with Loss Function and Optimizer
# Define the Mean Squared Error loss function and Stochastic Gradient Descent optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
 
# Step 3: Implement the Training Loop
# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass: compute predicted y by passing X to the model
    predictions = net(X)
    loss = loss_fn(predictions, Y)
 
    # Zero gradients, perform a backward pass, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
 
# Step 4: Visualize Final Predictions After Training and Compare with Initial Predictions
# Predictions after training
with torch.no_grad():
    final_predictions = net(X)
 
# Visualization of predictions before and after training
plt.figure(figsize=(10, 5))
 
# Before training
plt.subplot(1, 2, 1)
sns.lineplot(x=X.squeeze().numpy(), y=initial_predictions.squeeze().numpy())
plt.title('Initial Predictions (Before Training)')
plt.xlabel('Input X')
plt.ylabel('Predicted Y')
 
# After training
plt.subplot(1, 2, 2)
sns.lineplot(x=X.squeeze().numpy(), y=final_predictions.squeeze().numpy())
plt.title('Predictions After Training')
plt.xlabel('Input X')
plt.ylabel('Predicted Y')
 
plt.tight_layout()
plt.show()