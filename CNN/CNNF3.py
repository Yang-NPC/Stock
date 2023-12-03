from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
import numpy as np
def preprocess_data(file_path, time_step=100, test_size=0.2):
    data = pd.read_csv(file_path)
    selected_features = ['Close', 'Volume']
    data = data[selected_features]
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
    data.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = create_dataset(scaled_data, time_step)

    X_torch = torch.from_numpy(X).float().permute(0, 2, 1)
    y_torch = torch.from_numpy(y).float()

    X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size=test_size, random_state=42)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    return train_loader, test_loader, y_train, y_test

def evaluate_model(model, test_loader, criterion, y_test):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    rmse = np.sqrt(avg_test_loss)
    avg_close_price = y_test.mean().item()  # Use the passed y_test for calculation
    error_percentage = (rmse / avg_close_price) * 100

    return rmse, avg_close_price, error_percentage
# Function to create windowed dataset
def create_dataset(dataset, time_step=100):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        X.append(a)
        y.append(dataset[i + time_step, 0])  # Target is 'Close' price
    return np.array(X), np.array(y)

time_step = 100
time_step = 100
class StockPriceCNN(nn.Module):
    def __init__(self):
        super(StockPriceCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * ((time_step - 1) // 2), 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = StockPriceCNN()

file_paths = ['Stock\DataAquire\StockData\MinuteWise\INFY.NS.csv', 'Stock\DataAquire\StockData\MinuteWise\TCS.NS.csv', 'Stock\DataAquire\StockData\MinuteWise\CIPLA.NS.csv']

# Initialize the model
model = StockPriceCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Iterate through each file
for file_path in file_paths:
    # Preprocess the data
    train_loader, test_loader, y_train, y_test = preprocess_data(file_path)


    # Train the model
    # Define training parameters
    epochs = 100

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    rmse, avg_close_price, error_percentage = evaluate_model(model, test_loader, criterion, y_test)
    print(f"File: {file_path}, RMSE: {rmse}, Average Close Price: {avg_close_price}, Error Percentage: {error_percentage}%")