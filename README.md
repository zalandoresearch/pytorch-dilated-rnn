# PyTorch Dilated Recurrent Neural Networks

PyTorch implementation of  [Dilated Recurrent Neural Networks](https://arxiv.org/abs/1710.02224) (DilatedRNN).

## Getting Started

Installation:
```
pip install -r requirements.txt
```

Run the tests:
```
python tests.py
```

## Example

Define a dilated RNN based on GRU cells with 9 layers, dilations 1, 2, 4, 8, 16, ...
Then pass the hidden state to a further update
```python
import drnn
import torch

n_input = 20
n_hidden = 32
n_layers = 9
cell_type = 'GRU'

model = drnn.DRNN(n_input, n_hidden, n_layers, cell_type)

x1 = torch.autograd.Variable(torch.randn(23, 2, n_input))
x2 = torch.autograd.Variable(torch.randn(23, 2, n_input))

out, hidden = model(x1)
out, hidden = model(x2, hidden)
```

## Copy Task

```
python3 -m copy_memory.copymem_test --help
```
