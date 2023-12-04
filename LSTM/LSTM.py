from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch.optim as optim
import numpy as np

# Re-loading the dataset as the previous session state was reset
file_path = 'Stock\DataAquire\StockData\MinuteWise\CIPLA.NS.csv'
data = pd.read_csv(file_path)

# Selecting only the 'Close' (price) and 'Volume' columns
selected_features = ['Close', 'Volume']
data = data[selected_features]

# Convert columns to float
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

# Drop rows with NaN values (if any)
data.dropna(inplace=True)

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create windowed dataset
def create_dataset(dataset, time_step=100):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        X.append(a)
        y.append(dataset[i + time_step, 0])  # Target is 'Close' price
    return np.array(X), np.array(y)

time_step = 100
X, y = create_dataset(scaled_data, time_step)

# Display the first few rows of the preprocessed data
data.head(), X.shape, y.shape

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create windowed dataset
X, y = create_dataset(scaled_data, time_step=100)

# Convert to PyTorch tensors and permute
X_torch = torch.from_numpy(X).float().permute(0, 2, 1)
y_torch = torch.from_numpy(y).float()

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size=0.2, random_state=42)

# Create DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(out[:, -1, :])
        out = self.fc1(out)
        return out

model = StockPriceLSTM()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluate the model on the test dataset
model.eval()
test_loss = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        test_loss += loss.item()

# Calculate the average loss (MSE) on the test dataset
avg_test_loss = test_loss / len(test_loader)

# Calculate RMSE
rmse = np.sqrt(avg_test_loss)

# Calculate the average 'Close' price in the test dataset for error percentage
avg_close_price = y_test.mean().item()

# Calculate Error Percentage
error_percentage = (rmse / avg_close_price) * 100

print(f'RMSE: {rmse}, Average Close Price: {avg_close_price}, Error Percentage: {error_percentage}%')
