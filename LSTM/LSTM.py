from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch.optim as optim
import numpy as np

file_path = 'Stock\DataAquire\StockData\MinuteWise\INFY.NS.csv'
data = pd.read_csv(file_path)
data = data[:-1]

data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Close'])
close_prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
features_tensor = torch.FloatTensor(normalized_prices)

def create_dataset(data, length):
    seq = []
    L = len(data)
    for i in range(L-length):
        train_seq = data[i:i+length]
        train_label = data[i+length:i+length+1]
        seq.append((train_seq, train_label))
    return seq

seq_length = 3
train_size = int(len(features_tensor) * 0.8)

train_data = features_tensor[:train_size]
test_data = features_tensor[train_size:]

train_sequences = create_dataset(train_data, seq_length)
test_sequences = create_dataset(test_data, seq_length)

train_loader = DataLoader(train_sequences, batch_size=3, shuffle=True)
test_loader = DataLoader(train_sequences, batch_size=3, shuffle=False)
print(len(test_loader))

class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=100, num_layers=1):
        super(StockPriceLSTM, self).__init__()
        self.hidden_layer_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_layers)
        self.hidden_cell = torch.zeros(1, 1, self.hidden_layer_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(out[:, -1, :])
        out = self.fc1(out)
        return out

model = StockPriceLSTM()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        model.hidden_cell = torch.zeros(1, 1, model.hidden_layer_size)

        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1))

        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch:3}, Loss: {loss.item():10.8f}')

model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predictions.extend(y_test_pred.numpy().flatten().tolist())
        actuals.extend(labels.numpy().flatten().tolist())

actual_prices = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

min_length = min(len(actual_prices), len(predicted_prices))
actual_prices = actual_prices[:min_length]
predicted_prices = predicted_prices[:min_length]

results_df = pd.DataFrame({
    'Actual Price': np.squeeze(actual_prices),
    'Predicted Price': np.squeeze(predicted_prices)
})

percentage_errors = np.abs((actual_prices - predicted_prices) / actual_prices) * 100

average_percentage_error = np.mean(percentage_errors)
print(f"Average Percentage Error: {average_percentage_error:.2f}%")


plt.figure(figsize=(12, 6))
plt.plot(results_df['Actual Price'], label='Actual Price')
plt.plot(results_df['Predicted Price'], label='Predicted Price', alpha=0.7)
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
