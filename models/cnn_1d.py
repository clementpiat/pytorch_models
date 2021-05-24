from torch import nn
import torch

class CNN1D(nn.Module):
    def __init__(self, input_sample, kernel_size=3, channels=[32,32], hidden_dim=128, out_dim=1, dropout_rate=0):
        """
        input_sample: tensor of shape (1,input_length)
        """
        super(CNN1D, self).__init__()
        self.out_dim = out_dim

        self.cnn = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size, padding=1),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(channels[0], channels[1], kernel_size, padding=1),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.MaxPool1d(3)
        )
        
        self.cnn_out_dim = len(self.cnn(input_sample.unsqueeze(0)).flatten())

        self.mlp = nn.Sequential(
            nn.Linear(self.cnn_out_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, out_dim)
        )
        
    def forward(self, x):
        """
        x: tensor of shape (1,1,input_length)
        """
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        return self.mlp(x)
