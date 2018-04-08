from torch import nn

from drnn import DRNN


class DRNN_Char(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout=0.2, emb_dropout=0.2):
        super(DRNN_Char, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.drnn = DRNN(cell_type='QRNN',
                         dropout=dropout,
                         n_hidden=hidden_size,
                         n_input=input_size,
                         n_layers=num_layers,
                         batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        y, _ = self.drnn(emb)
        o = self.decoder(y)
        return o.contiguous()