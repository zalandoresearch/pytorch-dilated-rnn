import unittest
import drnn
import torch


class TestForward(unittest.TestCase):
    def test(self):
        model = drnn.DRNN(10, 10, 4, 0, 'GRU')

        x = torch.autograd.Variable(torch.randn(23, 3, 10))

        out = model(x)[0]

        self.assertTrue(out.size(0) == 23)
        self.assertTrue(out.size(1) == 3)
        self.assertTrue(out.size(2) == 10)


class TestReshape(unittest.TestCase):
    def test(self):
        model = drnn.DRNN(10, 10, 4, 0, 'GRU')

        x = torch.autograd.Variable(torch.randn(24, 3, 10))

        split_x = model._prepare_inputs(x, 2)

        second_block = x[1::2]
        check = split_x[:, x.size(1):, :]

        self.assertTrue((second_block == check).all())

        unsplit_x = model._split_outputs(split_x, 2)

        self.assertTrue((x == unsplit_x).all())


class TestHidden(unittest.TestCase):
    def test(self):
        model = drnn.DRNN(10, 10, 4, 0, 'GRU')

        x = torch.autograd.Variable(torch.randn(23, 3, 10))

        hidden = model(x)[1]

        self.assertEqual(len(hidden), 4)

        for hid in hidden:
            print(hid.size())

class TestQRNN(unittest.TestCase):
    def test(self):
        model = drnn.DRNN(10, 10, 4, 0, 'QRNN')

        x = torch.autograd.Variable(torch.randn(23, 3, 10))

        hidden = model(x)[1]

        self.assertEqual(len(hidden), 4)

        for hid in hidden:
            print(hid.size())


class TestPassHidden(unittest.TestCase):
    def test(self):
        model = drnn.DRNN(10, 10, 4, 0, 'GRU')

        hidden = []
        for i in range(4):
            hidden.append(torch.autograd.Variable(torch.randn(2 ** i, 3, 10)))

        x = torch.autograd.Variable(torch.randn(24, 3, 10))
        hidden = model(x, hidden)


if __name__ == '__main__':
    unittest.main()
