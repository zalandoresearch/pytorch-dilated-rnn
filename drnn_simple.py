import torch


V = torch.autograd.Variable


class DRNN(torch.nn.Module):
    def __init__(self, ninp, nhid, nlayers):

        torch.nn.Module.__init__(self)

        self.dilations = [2 ** i for i in range(nlayers)]

        self.cells = torch.nn.ModuleList([])

        for i in range(nlayers):
            self.cells.append(torch.nn.RNN(ninp, nhid, 1))
            ninp = nhid

    def layer(self, cell, x, r):

        x = torch.cat([x[i::r] for i in range(r)], 1)
        x = cell(x)[0]

        d = x.size(1) // r
        out = V(torch.zeros(x.size(0) * r, d, x.size(2)))

        for i in range(r):
                out[i::r] = x[:, i * d: (i + 1) * d, :]

        return out

    def forward(self, x):

        for cell, r in zip(self.cells, self.dilations):
            x = self.layer(cell, x, r)

        return x
