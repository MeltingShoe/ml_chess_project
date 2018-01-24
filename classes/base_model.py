import os
import torch.optim as optim
import torch
import torch.nn as nn
from torch.autograd import Variable

#This is the base class for models
class Network():
    """
    params:
        model: the model of type Model we want to use
        loss_function: a loss function like nn.MSELoss
        optimizer: an optimizer like optim.SGD
        learning rate: a float representing learning rate of the model
        use_cuda: a boolean whether to use CUDA, only works if CUDA is installed
        resume: flag to say if we are resuming training
        filepath: file where to save the model
    """
    def __init__(self, model, loss_function, optimizer, learning_rate, use_cuda=True, resume=False, filepath='auto'):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.filepath = filepath
        self.model = model
        if use_cuda:
            self.model.cuda()
        self.loss_function = loss_function()
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        if resume:
            if not self.load_checkpoint(): #TODO decide a pattern for filenames
                self.start_epoch = 0
                self.initialize_weights()
        else:
            self.start_epoch = 0
            self.initialize_weights()

    def training_session(self, train_data, test_data, n_epochs):
        """function to manage the train session"""
        for epoch in range(self.start_epoch, n_epochs, 1):
            self.train(train_data, epoch)
            self.validate(test_data)

        self.save_params()
        
    
    def train(self, dataloader, epoch, starting_index=0, print_batch=False):
        """function to train the network """
        epoch_loss = 0.0
        for i, data in enumerate(dataloader, starting_index):
            inputs, labels = data #get inputs and labels
            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            inputs = Variable(inputs)
            labels = Variable(labels)

            self.optimizer.zero_grad() #zero parameters gradient

            outputs = self.model(inputs) #forward
            loss = self.loss_function(outputs, labels) #compute loss
            loss.backward() #backpropagation
            self.optimizer.step() #optimization

            epoch_loss += loss.data[0]

            if print_batch:
                print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch + 1, i + 1, loss.data[0]))

            #save at the end of every epoch
            #we can save how many information as we want
            checkpoint =  {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
            }
            self.save_checkpoint(checkpoint)
        
    def validate(self, dataloader):
        correct_count = 0
        total_count = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            inputs = Variable(inputs)

            outputs = self.model(inputs)
            max, argmax = torch.max(outputs.data, 1)
            predicted = argmax

            correct_count += (predicted == labels).sum()
            total_count += dataloader.batch_size

        print('Accuracy on the validation set: {0}'.format(100.0 * correct_count / total_count))

    def initialize_weights(self, mean=0.0, variance=0.1, bias=0):
        """ intialize network weights as Normal Distribution of given mean and variance """
        for module in self.model.modules():
            if hasattr(module, "weight"):
                module.weight.data.normal_(mean, variance)
            if hasattr(module, "bias"):
                module.bias.data.fill_(bias)

    def save_params(self):
        """save model to file """
        if self.filepath == 'auto':
            path = 'path_to_file'
        else:
            path = self.filepath
        torch.save(self.model.state_dict(), path)

    def load_params(self):
        """ load model from file """
        if self.filepath == 'auto':
            path = 'path_to_file'
        else:
            path = self.filepath
        self.model.load_state_dict(torch.load(path))

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    def load_checkpoint(self, filename='checkpoint.pth.tar'):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
            return True
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            return False
