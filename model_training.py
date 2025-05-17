from model_creation import model
from torch.utils.data import DataLoader
from model_creation import train_dataset,validation_dataset
import torch.nn as nn
import torch
from togpu import device

# Define the loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=0.01)
# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(dataset=validation_dataset, batch_size=100, shuffle=False)


import time
import torch.nn.functional as F

n_epochs = 100
loss_list = []
accuracy_list = []
correct = 0

N_test = len(validation_dataset)     # assuming validation dataset is named val_dataset
N_train = len(train_dataset)
start_time = time.time()

Loss = 0
start_time = time.time()

for epoch in range(n_epochs):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        model.train()
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Make a prediction
        z = model(x)
        
        # Calculate loss
        loss = criterion(z, y)
        
        # Backpropagation
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Store the loss
        loss_list.append(loss.item())

    correct = 0
    for x_test, y_test in val_loader:
        # Set model to evaluation mode
        model.eval()
        x_test, y_test = x_test.to(device), y_test.to(device)
        
        # Disable gradient calculation
        with torch.no_grad():
            z = model(x_test)
            
            # Get predicted class (index of max log-probability)
            yhat = torch.argmax(z, dim=1)
            
            # Count correct predictions
            correct += (yhat == y_test).sum().item()

    accuracy = correct / N_test
    accuracy_list.append(accuracy)
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
