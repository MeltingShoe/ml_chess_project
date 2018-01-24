'''
This is a utilities module for any methods often used by other modules
Things like saving models and parsing board states go here
'''
import torch
import os


class utils():

    def save_params(self, filepath):
        """save model to file """
        if filepath == 'auto':
            path = 'path_to_file'
        else:
            path = filepath
        torch.save(self.model.state_dict(), path)

    def load_params(self, filepath):
        """ load model from file """
        if filepath == 'auto':
            path = 'path_to_file'
        else:
            path = filepath
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

    def initialize_weights(self, mean=0.0, variance=0.1, bias=0):
        """ intialize network weights as Normal Distribution of given mean and variance """
        for module in self.model.modules():
            if hasattr(module, "weight"):
                module.weight.data.normal_(mean, variance)
            if hasattr(module, "bias"):
                module.bias.data.fill_(bias)
