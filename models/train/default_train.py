import classes.utils as utils

from torch.autograd import Variable


class default_train():

    def __init__(self, loss_function, optimizer, trainable_params, learning_rate):
        self.loss_function = loss_function()
        self.optimizer = optimizer(trainable_params, lr=learning_rate)
        self.use_cuda = False

    def cuda(self):
        self.use_cuda = True

    def train(self, feed_forward, dataloader, epoch, starting_index=0, print_batch=False):
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
        utils.save_checkpoint(checkpoint)