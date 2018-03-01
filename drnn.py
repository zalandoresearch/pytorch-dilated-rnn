import torch
import torch.nn as nn
import torch.autograd as autograd


use_cuda = torch.cuda.is_available()


class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, cell_type='GRU'):

        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        self.cell_type = cell_type

        self.cells = torch.nn.ModuleList([])

        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden)
            else:
                c = cell(n_hidden, n_hidden)
            self.cells.append(c)

    def forward(self, inputs, hidden=None):

        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is not None:
                inputs = self.drnn_layer(cell, inputs, dilation, hidden=hidden[i])
                return inputs
            else:
                inputs = self.drnn_layer(cell, inputs, dilation)
                return inputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):

        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        dilated_outputs = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):

        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs = cell(dilated_inputs, hidden)[0]

        return dilated_outputs

    def _unpad_outputs(self, splitted_outputs, n_steps):

        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):

        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):

        iseven = (n_steps % rate) == 0

        if not iseven:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs, autograd.Variable(zeros_)))

        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):

        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)

        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        c = autograd.Variable(torch.zeros(batch_size, hidden_dim))
        if use_cuda:
            c = c.cuda()
        if self.cell_type == "LSTM":
            m = autograd.Variable(torch.zeros(batch_size, hidden_dim))
            if use_cuda:
                m = m.cuda()
            return (c, m)
        else:
            return c
