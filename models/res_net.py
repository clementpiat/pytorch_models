from torch import nn
import torch

class ResNet(nn.Module):
    """
    Implementation of a ResNet (for 1D sequences)
    """
    def __init__(self, input_sample, out_dim=1):
        """
        input_sample: tensor of shape (1,input_length)
        """
        super(ResNet, self).__init__()
        self.out_dim = out_dim
        self.input_length = input_sample.shape[-1]

        self.res_blocks = nn.ModuleList([self.get_res_block(first=not(i)) for i in range(3)])

        self.gap = nn.AvgPool1d(self.input_length)
        self.linear = nn.Linear(64,out_dim)

    def get_res_block(self, first=False):
        input_channel = 1 if first else 64

        return nn.Sequential(
            nn.Conv1d(input_channel, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        input_sample: tensor of shape (1,input_length)
        """
        x = self.res_blocks[0](x) + torch.cat([x]*64, dim=1)
        for res_block in self.res_blocks[1:]:
            x = res_block(x) + x

        x = torch.squeeze(self.gap(x), -1)
        return self.linear(x)
