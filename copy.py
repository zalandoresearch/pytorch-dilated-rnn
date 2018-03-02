import numpy
import torch
import drnn
import os

V = torch.autograd.Variable
use_cuda = torch.cuda.is_available()


def toydata(T, n):
    x = numpy.zeros((T + 20, n))

    x[:10] = numpy.random.randint(0, 8, size=[10, n])
    x[10:T + 9] = 8
    x[T + 9:] = 9

    return torch.from_numpy(x.astype(int))


class Copy(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.embed = torch.nn.Embedding(10, 10)

        self.drnn = drnn.DRNN(10, 10, 9, 'RNN')

        self.project = torch.nn.Linear(10, 8)

    def forward(self, input):
        embedded  = self.embed(input)

        hidden = self.drnn(embedded)[-10:]
        hidden = torch.cat([x.unsqueeze(0) for x in hidden], 0)
        return self.project(hidden)


class CopyBaseline(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.embed = torch.nn.Embedding(10, 10)

        self.drnn = torch.nn.GRU(10, 128, 1)

        self.project = torch.nn.Linear(128, 8)

    def forward(self, input):
        return self.drnn(self.embed(input))[0][-10:]


model = Copy()
if use_cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
it = 0

while True:

    batch = toydata(500, 128)

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

    print("echo '{},{}' >> log.csv".format(it, loss.data[0]))
