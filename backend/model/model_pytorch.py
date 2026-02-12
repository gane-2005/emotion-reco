import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN_BiLSTM(nn.Module):
    def __init__(self, num_classes=7, input_size=128):
        super(EmotionCNN_BiLSTM, self).__init__()
        
        # CNN layers for spatial feature extraction from spectrograms
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)
        
        # BiLSTM for temporal sequence modeling
        # Assuming we pool once, sequence length is halved
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        
        # Fully connected layers for classification
        # bidirectional=True means hidden_size*2
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch, input_size, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Prepare for LSTM: (batch, seq_len, input_size_for_lstm)
        # Transpose to (batch, seq_len, 128)
        x = x.transpose(1, 2)
        
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Take the last hidden state for classification
        # Shape: (batch, hidden_size*2)
        last_hidden = lstm_out[:, -1, :]
        
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def get_model(num_classes=7, input_size=128):
    return EmotionCNN_BiLSTM(num_classes=num_classes, input_size=input_size)
