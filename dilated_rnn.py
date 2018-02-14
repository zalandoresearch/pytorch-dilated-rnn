import torch
import torch.nn as nn


class DilatedRNN(nn.Module):
    """
    Applies a multi-layer dilated RNN (drnn) to an input sequence.

    Each layer implements the following recurrence, where $r$ is the rate:

        $$h_{t} = f(x_t, h_{t - r})$$

    The recurrence type is given by the mode.

    Args:
        mode: RNN class to implement forward steps in each layer
        input_size: The number of expected features in the input x
        dilations: 
        hidden_sizes: The number of features in the hidden states h
        bias: If ``False``, then the layer does not use bias weights b_ih and b_hh.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each
            RNN layer except the last layer

    Inputs: input
        input: (seq_len, batch, input_size): tensor containing the features
                of the input sequence.

    Outputs: output, h_n
        output: (seq_len, batch, hidden_size * num_directions): tensor
                 containing the output features `(h_t)` from the last layer of the RNN,
                 for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
                 given as the input, the output will also be a packed sequence.
        h_n: hidden state at last time point

    Attributes:
        layers: list of rnn instances for each layer

    """
    def __init__(self, mode, input_size, dilations, hidden_sizes, dropout):
        super(DilatedRNN, self).__init__()

        assert len(hidden_sizes) == len(dilations)

        self.dilations = dilations
        self.layers = torch.nn.ModuleList([])
        next_input_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(mode(input_size=next_input_size, 
                                    hidden_size=hidden_size,
                                    dropout=dropout))
            next_input_size = hidden_size

    def _pad_inputs(self, inputs, rate):
        num_steps = len(inputs)
        if num_steps % rate:
            dilated_num_steps = num_steps // rate + 1
            zeros_tensor = torch.zeros(dilated_num_steps * rate - num_steps,
                                       inputs.size(1),
                                       inputs.size(2))
            if torch.cuda.is_available():
                zeros_tensor = zeros_tensor.cuda()
            zeros_tensor = torch.autograd.Variable(zeros_tensor)
            inputs = torch.cat((inputs, zeros_tensor))
        return inputs

    def _stack(self, x, rate):
        tostack = [x[i::rate] for i in range(rate)]
        stacked = torch.cat(tostack, 1)
        return stacked

    def _unstack(self, x, rate):
        return x.view(x.size(0) * rate, -1, x.size(2))

    def _dilated_RNN(self, cell, inputs, rate, hidden=None):
        padded_inputs = self._pad_inputs(inputs, rate)
        dilated_inputs = self._stack(padded_inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = cell(dilated_inputs)
        else:
            hidden = self._stack(hidden, rate)
            dilated_outputs, hidden = cell(dilated_inputs, hidden)

        outputs = self._unstack(dilated_outputs, rate)
        hidden = self._unstack(hidden, rate)

        return outputs[:inputs.size(0)], hidden

    def forward(self, x, hidden_states=[None]):
        last_hidden = []
        for cell, dilation, hidden in zip(self.layers, self.dilations, hidden_states):
            if hidden is None:
                x, h = self._dilated_RNN(cell, x, dilation)
            else:
                x, h = self._dilated_RNN(cell, x, dilation, hidden=hidden)
            last_hidden.append(h)
        return x, last_hidden

    def __repr__(self):
        out = "DRNN with dilations: {}\n and layers: \n".format(self.dilations)

        append_ = "\n".join([cell.__repr__() for cell in self.layers])

        return out + append_ + "\n"
