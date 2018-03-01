import torch
import torch.nn as nn
import torch.autograd as autograd


class multi_dRNN_with_dilations(nn.Module):

    def __init__(self, hidden_structs, dilations, input_dims, cell_type, use_cell=False):

        self.use_cell = use_cell

        print "using cell: {}".format(self.use_cell)

        super(multi_dRNN_with_dilations, self).__init__()
        
        self.hidden_structs = hidden_structs
        self.dilations = dilations
        self.cell_type = cell_type

        self.cells = torch.nn.ModuleList([])

        if self.use_cell:

            if self.cell_type == "LSTM":
                cell = nn.LSTMCell
            elif self.cell_type == "GRU":
                cell = nn.GRUCell
            elif self.cell_type == "RNN":
                cell = nn.RNNCell

        else:
            if self.cell_type == "GRU":
                cell = nn.GRU
            else:
                raise NotImplementedError

        lastHiddenDim = -1
        for i, hidden_dims in enumerate(hidden_structs):
            if i == 0:
                c = cell(input_dims, hidden_dims)
            else:
                c = cell(lastHiddenDim, hidden_dims)
            
            self.cells.append(c)
            lastHiddenDim = hidden_dims
    
    def forward(self, inputs):

        for cell, dilation in zip(self.cells, self.dilations):

            inputs = self.dRNN(cell, inputs, dilation)
            
        return inputs
    
    def dRNN(self, cell, inputs, rate):

        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)

        dilated_inputs = self._prepare_inputs(inputs, rate, dilated_steps)

        dilated_outputs = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)

        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size):
        dilated_outputs = []
        if self.cell_type == "LSTM":

            hidden, cstate = self.init_hidden(batch_size * rate, hidden_size)
            for dilated_input in dilated_inputs:
                hidden, cstate = cell(dilated_input, (hidden, cstate))
                dilated_outputs.append(hidden)

        else:
            if self.use_cell:
                hidden = self.init_hidden(batch_size * rate, hidden_size)

                for dilated_input in dilated_inputs:
                    hidden = cell(dilated_input, hidden)
                    dilated_outputs.append(hidden)
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

                dilated_outputs = cell(dilated_inputs, hidden)[0]

        return dilated_outputs

    def _unpad_outputs(self, splitted_outputs, n_steps):
        if self.use_cell:
            unrolled_outputs = [output
                                for sublist in splitted_outputs
                                for output in sublist]
            return  unrolled_outputs[:n_steps]
        else:
            return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        if self.use_cell:
            return [torch.chunk(output, rate, 0)
                    for output in dilated_outputs]
        else:
            output = torch.zeros(dilated_outputs.size(0) * rate,
                                 dilated_outputs.size(1) // rate,
                                 dilated_outputs.size(2))
            output = torch.autograd.Variable(output)

            batchsize = dilated_outputs.size(1) / rate

            blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :]
                      for i in range(rate)]

            interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
            interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                           batchsize,
                                           dilated_outputs.size(2))
            return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):

        iseven = (n_steps % rate) == 0

        if not iseven:
            dilated_steps = n_steps // rate + 1  # ceiling
            if self.use_cell:
                zero_tensor = autograd.Variable(inputs[0].data.new(inputs[0].data.size()).zero_())
                for i_pad in range(dilated_steps * rate - n_steps):
                    inputs.append(zero_tensor)
            else:
                zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                     inputs.size(1),
                                     inputs.size(2))
                inputs = torch.cat((inputs, autograd.Variable(zeros_)))

        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate, dilated_steps):
        if self.use_cell:
            dilated_inputs = [torch.cat([inputs[i * rate + j]
                                         for j in range(rate)], dim=0)
                              for i in range(dilated_steps)]
        else:
            dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)

        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        if self.cell_type == "LSTM":
            return (autograd.Variable(torch.zeros(batch_size, hidden_dim)),
                    autograd.Variable(torch.zeros(batch_size, hidden_dim)))
        else:
            return autograd.Variable(torch.zeros(batch_size, hidden_dim))


class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, cell_type='GRU'):

        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        self.cell_type = cell_type

        self.cells = torch.nn.ModuleList([])

        if self.cell_type == "GRU":
            cell = nn.GRU
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden)
            else:
                c = cell(n_hidden, n_hidden)
            self.cells.append(c)

    def forward(self, inputs):

        for cell, dilation in zip(self.cells, self.dilations):
            inputs = self.drnn_layer(cell, inputs, dilation)

        return inputs

    def drnn_layer(self, cell, inputs, rate):

        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        dilated_outputs = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size):

        hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)
        dilated_outputs = cell(dilated_inputs, hidden)[0]

        return dilated_outputs

    def _unpad_outputs(self, splitted_outputs, n_steps):

        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):

        batchsize = dilated_outputs.size(1) / rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :]
                  for i in range(rate)]

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
            inputs = torch.cat((inputs, autograd.Variable(zeros_)))

        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):

        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)

        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        if self.cell_type == "LSTM":
            return (autograd.Variable(torch.zeros(batch_size, hidden_dim)),
                    autograd.Variable(torch.zeros(batch_size, hidden_dim)))
        else:
            return autograd.Variable(torch.zeros(batch_size, hidden_dim))
