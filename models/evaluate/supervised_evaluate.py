import torch
from torch.autograd import Variable


class supervised_evaluate():

    def __init__(self):
        self.use_cuda = False

    def __call__(self, *args, **kwargs):
        self.evaluate(*args, **kwargs)

    def cuda(self):
        self.use_cuda = True

    def evaluate(self, feed_forward, dataloader):
        correct_count = 0
        total_count = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            inputs = Variable(inputs)

            outputs = feed_forward(inputs)
            max, argmax = torch.max(outputs.data, 1)
            predicted = argmax

            correct_count += (predicted == labels).sum()
            total_count += dataloader.batch_size

        print('Accuracy on the validation set: {0}'.format(
            100.0 * correct_count / total_count))