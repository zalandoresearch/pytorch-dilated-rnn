import numpy
import torch
import drnn_simple
import dilated_rnn
import os

V = torch.autograd.Variable
use_cuda = torch.cuda.is_available()


def toydata(T, n):
    x = numpy.zeros((T + 20, n))

    x[:10] = numpy.random.randint(0, 8, size=[10, n])
    x[10:T + 9] = 8
    x[T + 9:] = 9
    return torch.from_numpy(x.astype(int))


class CopySimple(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.embed = torch.nn.Embedding(10, 10)

        self.drnn = drnn_simple.DRNN(10, 32, 6)

        self.project = torch.nn.Linear(32, 8)

    def forward(self, input):
        return self.project(self.drnn(self.embed(input))[-10:])

    def cuda(self):
        self.embed.cuda()
        self.drnn.cuda()
        self.project.cuda()


class Copy(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.embed = torch.nn.Embedding(10, 10)

        self.drnn = dilated_rnn.DilatedRNN(
            torch.nn.RNN,
            10,
            [1, 2, 4, 8, 16],
            [128] * 5,
            0.0,
        )

        self.project = torch.nn.Linear(128, 8)

    def forward(self, input):
        return self.project(self.drnn(self.embed(input))[-10:])

    def cuda(self):
        self.embed.cuda()
        self.drnn.cuda()
        self.project.cuda()


class CopyBaseline(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.embed = torch.nn.Embedding(10, 10)

        self.drnn = torch.nn.GRU(10, 128, 1)

        self.project = torch.nn.Linear(128, 8)

    def forward(self, input):
        return self.drnn(self.embed(input))[0][-10:]

    def cuda(self):
        self.embed.cuda()
        self.drnn.cuda()
        self.project.cuda()


model = CopySimple()
if use_cuda:
    model.cuda()

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
criterion = torch.nn.CrossEntropyLoss()

os.system('echo "Iteration,Cross-Entropy" > log.csv')

it = 0

while True:

    batch = toydata(12, 128)

    if use_cuda:
        batch = batch.cuda()

    optimizer.zero_grad()

    output = model(V(batch))

    loss = 0
    for j in range(output.size(1)):
        loss += criterion(output[:, j, :], V(batch[:10, j]))
    loss = loss / output.size(1)

    loss.backward()
    optimizer.step()

    if it % 10 == 0:
        print(loss.data[0])

    it += 1

    os.system("echo '{},{}' >> log.csv".format(it, loss.data[0]))