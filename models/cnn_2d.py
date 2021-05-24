from torch import nn
import torch

class CNN2D(nn.Module):
    def __init__(self, input_sample, kernel_size=(3,3), channels=[32,64], hidden_dim=128, out_dim=1):
        """
        input_sample: tensor of shape (1,1,width,height), or (1,1,height,width)
        """
        super(CNN2D, self).__init__()
        self.out_dim = out_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size, padding=1),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], kernel_size, padding=1),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.cnn_out_dim = len(self.cnn(input_sample.unsqueeze(0)).flatten())
        print(f"CNN out dim: {self.cnn_out_dim}")

        self.mlp = nn.Sequential(
            nn.Linear(self.cnn_out_dim, hidden_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, out_dim)
        )
        
    def forward(self, x):
        """
        x: tensor of shape (1,1,width,height), or (1,1,height,width)
        """
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        return self.mlp(x)
