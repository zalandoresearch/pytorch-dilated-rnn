import unittest
import torch
from dilated_rnn import DilatedRNN
import random
import drnn_simple


use_cuda = torch.cuda.is_available()


class TestPadInputs(unittest.TestCase):
    def test(self):
        drnn = DilatedRNN(
            mode=torch.nn.GRU,
            input_size=13,
            dilations=[1, 2, 4, 8],
            hidden_sizes=[8, 16, 32, 64],
            dropout=0.5
        )

        x = torch.autograd.Variable(torch.randn(15, 2, 13))

        if use_cuda:
            x = x.cuda()
            drnn = drnn.cuda()

        for rate in [2, 8]:

            padded = drnn._pad_inputs(x, rate).data

            self.assertEqual(padded.size(0), 16)

        for rate in [3, 5]:

            padded = drnn._pad_inputs(x, rate).data

            self.assertEqual(padded.size(0), 15)

        for rate in [12]:

            padded = drnn._pad_inputs(x, rate).data

            self.assertEqual(padded.size(0), 24)

        for rate in [18]:

            padded = drnn._pad_inputs(x, rate).data

            self.assertEqual(padded.size(0), 18)

        self.assertEqual(padded[-1, 0, 0], 0)


class TestStackInputs(unittest.TestCase):
    def test(self):

        drnn = DilatedRNN(
            mode=torch.nn.GRU,
            input_size=13,
            dilations=[1, 2, 4, 8],
            hidden_sizes=[8, 16, 32, 64],
            dropout=0.5
        )

        x = torch.autograd.Variable(torch.randn(16, 2, 13))

        if use_cuda:
            x = x.cuda()
            drnn = drnn.cuda()

        chunked = drnn._stack(x, 4)

        self.assertEqual(chunked.size(0), 4)
        self.assertEqual(chunked.size(1), 8)
        self.assertEqual(chunked.size(2), 13)

        self.assertTrue(torch.equal(x[0::4, 0, :], chunked[:, 0, :]))
        self.assertTrue(torch.equal(x[1::4, 0, :], chunked[:, 2, :]))
        self.assertTrue(torch.equal(x[2::4, 0, :], chunked[:, 4, :]))


class TestUnstackInputs(unittest.TestCase):
    def test(self):

        drnn = DilatedRNN(
            mode=torch.nn.GRU,
            input_size=13,
            dilations=[1, 2, 4, 8],
            hidden_sizes=[8, 16, 32, 64],
            dropout=0.5
        )

        x = torch.autograd.Variable(torch.randn(16, 2, 13))

        if use_cuda:
            x = x.cuda()
            drnn = drnn.cuda()

        self.assertTrue(torch.equal(drnn._unstack(drnn._stack(x, 4), 4), x))


class TestForward(unittest.TestCase):
    def test(self):

        dilatations = [1, 2, 4, 8]
        batch_size = 2
        input_size = 13
        hidden_size = 32

        drnn = DilatedRNN(
            mode=torch.nn.GRU,
            input_size=13,
            dilations=dilatations,
            hidden_sizes=[hidden_size]*4,
            dropout=0.5
        )

        x = torch.autograd.Variable(torch.randn(15, batch_size, input_size))
        hidden_states = [
            torch.autograd.Variable(torch.randn(1, rate*batch_size , hidden_size))
            for rate in dilatations]

        if use_cuda:
            x = x.cuda()
            hidden_states = hidden_states.cuda()
            drnn = drnn.cuda()

        outputs, hidden_states = drnn(x, hidden_states)

        self.assertEqual(outputs.size(0), 15)
        self.assertEqual(outputs.size(1), batch_size)
        self.assertEqual(outputs.size(2), hidden_size)


class TestForwardSimple(unittest.TestCase):
    def test(self):

        drnn = drnn_simple.DRNN(13, 10, 8)

        x = torch.autograd.Variable(torch.randn(256, 2, 13))

        if use_cuda:
            x = x.cuda()
            drnn = drnn.cuda()

        outputs = drnn(x)

        print(outputs.size())


class TestReuse(unittest.TestCase):
    def test(self):
        drnn = DilatedRNN(
            mode=torch.nn.GRU,
            input_size=13,
            dilations=[1, 2, 4, 8],
            hidden_sizes=[8, 16, 32, 64],
            dropout=0.5
        )

        x = torch.autograd.Variable(torch.randn(15, 2, 13))

        if use_cuda:
            x = x.cuda()
            drnn = drnn.cuda()

        y = x.clone()

        drnn(x)

        self.assertTrue(torch.equal(x.data, y.data))


class ToyModel(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.embedding = torch.nn.Embedding(26, 26)
        self.drnn = DilatedRNN(
            torch.nn.GRU,
            26,
            [1, 2],
            hidden_sizes=[128, 128],
            dropout=0.0
        )
        self.project = torch.nn.Linear(128, 26)

    def forward(self, input):
        out, hidden = self.drnn(self.embedding(input))
        return self.project(out)


class ToyModelOther(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.embedding = torch.nn.Embedding(26, 26)
        self.drnn = torch.nn.GRU(26, 128)
        self.project = torch.nn.Linear(128, 26)

    def forward(self, input):
        out, _ = self.drnn(self.embedding(input))[0]
        return self.project(out)


class ToyModelSimple(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.embedding = torch.nn.Embedding(26, 26)
        self.drnn = drnn_simple.DRNN(26, 128, 3)
        self.project = torch.nn.Linear(128, 26)

    def forward(self, input):
        out = self.drnn(self.embedding(input))
        return self.project(out)


class TestLearn(unittest.TestCase):
    def test(self):

        data = []
        for i in range(1000):
            start = random.randint(0, 26)
            data.append([x % 26 for x in range(start, start + 65)])

        data = torch.LongTensor(data).transpose(1, 0)
        data = torch.autograd.Variable(data)

        model = ToyModel()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1):
            for i in range(100):

                batch = data[:, i * 10: (i + 1) * 10]

                optimizer.zero_grad()

                output = model(batch[:-1])

                loss = 0

                for j in range(output.size(1)):
                    loss += criterion(output[:, j, :], batch[1:, j])

                loss = loss / output.size(1)

                loss.backward()

                optimizer.step()

                if i % 10 == 0:
                    print(loss.data[0])

        self.assertTrue(loss.data[0] < 0.1)


if __name__ == "__main__":
    unittest.main()
