import classes.utils as utils
from torch.autograd import Variable
import torch


def default_train(self, feed_forward, dataloader,
                  epoch, starting_index=0, print_batch=False):
    """function to train the network """
    epoch_loss = 0.0
    for i, data in enumerate(dataloader, starting_index):
        inputs, labels = data  # get inputs and labels 
        if self.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs = Variable(inputs)
        labels = Variable(labels)

        self.optimizer.zero_grad()  # zero parameters gradient

        outputs = feed_forward(inputs)  # forward
        loss = self.loss_function(outputs, labels)  # compute loss
        loss.backward()  # backpropagation
        self.optimizer.step()  # optimization

        epoch_loss += loss.data[0]

        if print_batch:
            print('[{:d}, {:5d}] loss: {:.3f}'.format(
                epoch + 1, i + 1, loss.data[0]))

    # save at the end of every epoch
    # we can save how many information as we want
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': feed_forward.state_dict(),
        'optimizer': self.optimizer.state_dict(),
    }
    utils.save_checkpoint(checkpoint, self.name)


def chess_train(self, feed_forward, dataset,
                  epoch, starting_index=0, print_batch=False):
    """function to train the network """
    epoch_loss = 0.0
    dataloader = 'placeholder'
    for i, data in enumerate(dataloader, starting_index):
        inputs, labels = data  # get inputs and labels 
        if self.use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs = Variable(inputs)
        labels = Variable(labels)

        self.optimizer.zero_grad()  # zero parameters gradient

        outputs = feed_forward(inputs)  # forward
        loss = self.loss_function(outputs, labels)  # compute loss
        loss.backward()  # backpropagation
        self.optimizer.step()  # optimization

        epoch_loss += loss.data[0]

        if print_batch:
            print('[{:d}, {:5d}] loss: {:.3f}'.format(
                epoch + 1, i + 1, loss.data[0]))

    # save at the end of every epoch
    # we can save how many information as we want
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': feed_forward.state_dict(),
        'optimizer': self.optimizer.state_dict(),
    }
    utils.save_checkpoint(checkpoint, self.name)
