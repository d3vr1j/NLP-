#RNN

import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out

# Example
model = SimpleRNN(input_size=10, hidden_size=20, output_size=2)
x = torch.randn(5, 3, 10)  # (batch, seq_len, input_size)
output = model(x)
print(output)
