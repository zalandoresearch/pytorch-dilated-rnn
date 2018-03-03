import argparse
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import time
import drnn


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./coco/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='DRNN',
                    help='type of recurrent net')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=64,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=4,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='./model.pt',
                    help='path to save the final model')


args = parser.parse_args()


use_cuda = torch.cuda.is_available()
if use_cuda:
    print("=" * 2 + "> USING CUDA!!!")


class RNNModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        elif rnn_type == 'DRNN':
            self.rnn = drnn.DRNN(ninp, nhid, nlayers, 'GRU')

        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        try:
            self.init_weights()
        except AttributeError:
            pass

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)

        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for i, line in enumerate(f):
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def batchify(data, bsz, cuda):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data


def get_batch(source, i, bptt, evaluation=False):
    seq_len = min(bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(data_source):
    lm.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)

    if args.model == 'DRNN':
        hidden = [Variable(torch.zeros(2 ** i, eval_batch_size, args.nhid))
                  for i in range(args.nlayers)]
        if use_cuda:
            hidden = [x.cuda() for x in hidden]

    else:
        hidden = lm.init_hidden(eval_batch_size)

    iter_ = range(0, data_source.size(0) - 1, args.bptt)
    for i in iter_[:-1]:
        dat_, targets = get_batch(data_source, i, args.bptt, evaluation=True)
        output, hidden = lm(dat_, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(dat_) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train(lm, train_data):
    lm.train()

    total_loss = 0
    start_time = time.time()

    if args.model == 'DRNN':
        hidden = [Variable(torch.zeros(2 ** i, args.batch_size, args.nhid))
                  for i in range(args.nlayers)]
        if use_cuda:
            hidden = [x.cuda() for x in hidden]

    else:
        hidden = lm.init_hidden(args.batch_size)

    iter_ = range(0, train_data.size(0) - 1, args.bptt)
    for batch, i in enumerate(iter_[:-1]):

        dat_, targets = get_batch(train_data, i, args.bptt)

        hidden = repackage_hidden(hidden)
        lm.zero_grad()

        output, hidden = lm(dat_, hidden)
        loss = criterion(output.view(-1, len(corpus.dictionary)), targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm(lm.parameters(), args.clip)
        for p in lm.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()



if __name__ == '__main__':

    torch.manual_seed(args.seed)

    print("getting data...")
    corpus = Corpus(args.data)

    eval_batch_size = 10

    print("batching...")

    stops = [i for i in range(len(corpus.train))
             if corpus.train[i] == corpus.dictionary.word2idx["<eos>"]]

    train_data = batchify(corpus.train, args.batch_size, use_cuda)
    valid_data = batchify(corpus.valid, eval_batch_size, use_cuda)
    test_data = batchify(corpus.test, eval_batch_size, use_cuda)

    print("getting model...")

    ntokens = len(corpus.dictionary)
    lm = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)

    if use_cuda:
        lm.cuda()

    criterion = nn.CrossEntropyLoss()

    lr = args.lr
    best_val_loss = None

    print("training...")
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(lm, train_data)

            val_loss = evaluate(valid_data)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)

            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(lm, f)
                best_val_loss = val_loss
            else:
                lr /= 4.0

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    with open(args.save, 'rb') as f:
        lm = torch.load(f)

    test_loss = evaluate(test_data)

    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
