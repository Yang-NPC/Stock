
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Load and preprocess data
file_path = '../DataAquire/StockData/MinuteWise/TCS.NS.csv'
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
print(len(test_loader))

# RNN model definition
class RNN(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=100, output_size=1):
        super(RNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.rnn = nn.RNN(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = torch.zeros(1, 1, self.hidden_layer_size)

    def forward(self, input_seq):
        rnn_out, self.hidden_cell = self.rnn(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(rnn_out.view(len(input_seq), -1))
        return predictions[-1]

model = RNN()

# Training the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 150

for i in range(epochs):
    for seq, labels in train_loader:
        optimizer.zero_grad()
        model.hidden_cell = torch.zeros(1, 1, model.hidden_layer_size)

        # Correctly reshape the sequence
        #seq = seq.view(seq_length, -1, 1)

        y_pred = model(seq)

        single_loss = criterion(y_pred, labels.view(-1))
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


plt.figure(figsize=(12, 6))
plt.plot(results_df['Actual Price'], label='Actual Price')
plt.plot(results_df['Predicted Price'], label='Predicted Price', alpha=0.7)
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()



