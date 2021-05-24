from torch import nn
import torch
import numpy as np

class AttentionModule(nn.Module):
    def __init__(self, input_length, n_channels, nheads):
        super(AttentionModule, self).__init__()

        self.K = nn.Linear(n_channels, n_channels)
        self.Q = nn.Linear(n_channels, n_channels)
        self.V = nn.Linear(n_channels, n_channels)

        self.attention = nn.MultiheadAttention(n_channels, nheads)

    def get_positional_encoding(self, n_batches, input_length, n_channels):
        """
        Not used
        """
        def encoding(i,t):
            w = 1e-4 ** (2 * (i//2) / n_channels)
            return np.cos(w*t) if i%2 else np.sin(w*t)

        p = torch.from_numpy(np.array([[encoding(i,t) for i in range(n_channels)] for t in range(input_length)]))
        return torch.stack([p]*n_batches, dim=0)

    def forward(self, x, with_weights=False):
        """
        x : tensor of shape (n_batches, n_channels, input_length)
        with_weight : if True returns the attention weights
        """
        x = x.transpose(1,2) # shape (n_batches, input_length, n_channels)

        key = self.K(x).transpose(0,1) # shape (input_length, n_batches, n_channels)
        query = self.Q(x).transpose(0,1)
        value = self.V(x).transpose(0,1)

        attn_output, attn_weights = self.attention(query, key, value)
        attn_output = attn_output.transpose(0,1).transpose(1,2) # shape (n_batches, n_channels, input_length)

        output = attn_output + x.transpose(1,2)
        if with_weights:
            return output, attn_weights

        return output

class AttentionCNN1D(nn.Module):    
    """
    Implementation of a CNN with attention modules (for sequences)
    """
    def __init__(self, input_sample, kernel_sizes=[7, 7, 7], channels=[32, 32, 32], strides=[3,2,1], hidden_dim=64, out_dim=1):
        """
        input_sample : tensor of shape (1,input_length)
        """
        super(AttentionCNN1D, self).__init__()
        self.out_dim = out_dim
        input_sample = input_sample.unsqueeze(0)

        self.cnn1, self.attention_module1 = self.get_block(1,channels[0], strides[0], input_sample)
        input_sample2 = self.attention_module1(self.cnn1(input_sample))
        self.cnn2, self.attention_module2 = self.get_block(channels[0],channels[1], strides[1], input_sample2)
        input_sample3 = self.attention_module2(self.cnn2(input_sample2))
        self.cnn3, self.attention_module3 = self.get_block(channels[1],channels[2], strides[2], input_sample3)

        self.cnn = nn.Sequential(
            self.cnn1,
            self.attention_module1,
            self.cnn2,
            self.attention_module2,
            self.cnn3,
            self.attention_module3,
        )

        output_sample = self.cnn(input_sample)
        self.cnn_out_dim = len(output_sample.flatten())
        print(f"CNN out dim : {self.cnn_out_dim}")
        self.mlp = nn.Sequential(
            nn.Linear(self.cnn_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, out_dim)
        )

    def get_block(self, input_channels, output_channels, stride, input_sample, dilation=1, max_pooling=3, nheads=4):
        cnn = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, output_channels, stride=stride, dilation=dilation),
            nn.ReLU(),
            nn.MaxPool1d(max_pooling)
        )

        input_length =  cnn(input_sample).shape[-1]
        attention_module = AttentionModule(input_length, output_channels, nheads)
        return cnn, attention_module
        
    def forward(self, x):
        """
        x : tensor of shape (1,input_length)
        """
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        return self.mlp(x)
