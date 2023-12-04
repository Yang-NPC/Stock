import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load and preprocess data
file_path = 'Stock\DataAquire\StockData\MinuteWise\CIPLA.NS.csv'
data = pd.read_csv(file_path)
data = data[:-1]
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Close'])
close_prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
normalized_prices = scaler.fit_transform(close_prices)
features_tensor = torch.FloatTensor(normalized_prices)

# Function to create sequences
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

seq_length = 3
train_size = int(len(features_tensor) * 0.8)

train_data = features_tensor[:train_size]
test_data = features_tensor[train_size:]

train_sequences = create_inout_sequences(train_data, seq_length)
test_sequences = create_inout_sequences(test_data, seq_length)

train_loader = DataLoader(train_sequences, batch_size=3, shuffle=True)
test_loader = DataLoader(test_sequences, batch_size=3, shuffle=False)

# CNN model definition
class CNN(nn.Module):
    def __init__(self, input_size=1, num_filters=32, filter_size=2, output_size=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=filter_size)
        self.fc1 = nn.Linear(num_filters * (seq_length - filter_size + 1), output_size)

    def forward(self, input_seq):
        # input_seq shape: (batch_size, seq_length, input_size)
        # transpose to (batch_size, input_size, seq_length) for Conv1d
        conv_out = self.conv1(input_seq.transpose(1, 2))
        # Make the tensor contiguous before calling view
        flattened = conv_out.contiguous().view(conv_out.size(0), -1)
        predictions = self.fc1(flattened)
        return predictions

model = CNN()

# Training the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 150

for i in range(epochs):
    for seq, labels in train_loader:
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = criterion(y_pred.view(-1), labels.view(-1))
        single_loss.backward()
        optimizer.step()

    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

# Evaluating the model
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for seq, labels in test_loader:
        y_test_pred = model(seq)
        predictions.extend(y_test_pred.numpy().flatten().tolist())  # Flatten and to list
        actuals.extend(labels.numpy().flatten().tolist())  # Flatten and to list

actual_prices = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

min_length = min(len(actual_prices), len(predicted_prices))
actual_prices = actual_prices[:min_length]
predicted_prices = predicted_prices[:min_length]

# Create a DataFrame
results_df = pd.DataFrame({
    'Actual Price': np.squeeze(actual_prices),
    'Predicted Price': np.squeeze(predicted_prices)
})

# Calculate percentage error for each prediction
percentage_errors = np.abs((actual_prices - predicted_prices) / actual_prices) * 100

# Now you can display the average percentage error over the test set
average_percentage_error = np.mean(percentage_errors)
print(f"Average Percentage Error: {average_percentage_error:.2f}%")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(results_df['Actual Price'], label='Actual Price')
plt.plot(results_df['Predicted Price'], label='Predicted Price', alpha=0.7)
plt.title('Stock Price Prediction with CNN')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
