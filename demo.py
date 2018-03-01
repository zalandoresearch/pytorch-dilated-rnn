import torch
import torchvision
import torch.autograd as autograd
import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
import drnn


class Classifier(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_layers, n_classes, cell_type="GRU"):
        super(Classifier, self).__init__()

        self.drnn = drnn.DRNN(n_inputs, n_hidden, n_layers, cell_type=cell_type)
        self.linear = nn.Linear(n_hidden, n_classes)

    def forward(self, inputs):
        layer_outputs = self.drnn(inputs)
        pred = self.linear(layer_outputs[-1])

        return pred


if __name__ == '__main__':

    data_dir = '.MNIST_data'
    n_classes = 10

    cell_type = "GRU"
    n_hidden = 20
    n_layers = 4

    batch_size = 128
    learning_rate = 1.0e-3
    training_iters = 30000
    display_step = 1

    train_data = torchvision.datasets.MNIST(root=data_dir,
                                            train=True,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True
                                            )


    test_data = torchvision.datasets.MNIST(root=data_dir,
                                           train = False
                                           )

    test_x = autograd.Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000] / 255.0
    test_x = test_x.view(test_x.size(0), 784).unsqueeze(2).transpose(1, 0)

    test_y = test_data.test_labels[:2000]

    train_loader = Data.DataLoader(train_data, batch_size, shuffle=False, num_workers=1)

    print("==> Building a dRNN with %s cells" %cell_type)

    model = Classifier(1, n_hidden, n_layers, n_classes, cell_type=cell_type)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for iter in range(training_iters):
        for step, (batch_x, batch_y) in enumerate(train_loader):

            batch_x = autograd.Variable(batch_x.view(-1, 28, 28))
            batch_x = batch_x.view(batch_size, 784).unsqueeze(2).transpose(1, 0)
            batch_y = autograd.Variable(batch_y)

            optimizer.zero_grad()

            pred = model.forward(batch_x)

            loss = criterion(pred, batch_y)

            loss.backward()
            optimizer.step()

            if (step + 1) % display_step == 0:
                print("Iter " + str(iter + 1) + ", Step " + str(step+1) +", Average Loss: " + "{:.6f}".format(loss.data[0]))

        test_output = model.forward(test_x)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        accuracy = sum(pred_y == test_y) / float(test_y.size(0))

        print("========> Validation Accuracy: {:.6f}".format(accuracy))

    print("end")


