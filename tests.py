import unittest
import drnn
import torch


class Test(unittest.TestCase):
    def test(self):
        model = drnn.multi_dRNN_with_dilations(
            [10, 10, 10, 10, 10],
            [1, 2, 4, 8, 16],
            10,
            'GRU',
            use_cell=False
        )

        x = torch.autograd.Variable(torch.randn(23, 3, 10))

        out = model(x)

        print out


if __name__ == '__main__':
    unittest.main()