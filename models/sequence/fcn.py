from torch import nn
import torch

class FCN(nn.Module):
    """
    Implementation of a Fully Convolutional Network (for 1D sequences)
    """
    def __init__(self, input_sample, out_dim=1):
        """
        input_sample: tensor of shape (1,input_length)
        """
        super(FCN, self).__init__()
        self.out_dim = out_dim

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 128, 8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
        )

        self.hidden_dim = self.cnn(torch.unsqueeze(input_sample,0)).shape[-1]

        self.gap = nn.AvgPool1d(self.hidden_dim)
        self.linear = nn.Linear(128, out_dim)


    def forward(self, x):
        """
        x: tensor of shape (1,1,input_length)
        """
        x = self.cnn(x)
        x = torch.squeeze(self.gap(x), -1)
        return self.linear(x)
