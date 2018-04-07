from torch import nn

from drnn import DRNN


class DRNN_Copy(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(DRNN_Copy, self).__init__()
        self.drnn = DRNN(cell_type='GRU', dropout=dropout, n_hidden=hidden_size,
                         n_input=input_size, n_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0,0.01)

    def forward(self, x): # x: (batch, steps, input_size)
        y1, _ = self.drnn(x) # y1: (batch, steps, hidden_size)
        #import pdb
        #pdb.set_trace()
        return self.linear(y1) # (batch, steps, output_size)