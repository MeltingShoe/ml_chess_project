'''
This is a utilities module for any methods often used by other modules
Things like saving models and parsing board states go here
'''
import torch
import os


def save_params(state_dict, filepath):
    """save model to file """
    if filepath == 'auto':
        path = 'path_to_file'
    else:
        path = filepath
    torch.save(state_dict, path)


def load_params(feed_forward, filepath):
    """ load model from file """
    if filepath == 'auto':
        path = 'path_to_file'
    else:
        path = filepath
    feed_forward.load_state_dict(torch.load(path))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_checkpoint(model, filename='checkpoint.pth.tar'):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.start_epoch = checkpoint['epoch']
        model.feed_forward.load_state_dict(checkpoint['state_dict'])
        model.training_method.optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
        return True
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        return False


def initialize_weights(modules, mean=0.0, variance=0.1, bias=0):
    """ intialize network weights as Normal Distribution of given mean and variance """
    for module in modules:
        if hasattr(module, "weight"):
            module.weight.data.normal_(mean, variance)
        if hasattr(module, "bias"):
            module.bias.data.fill_(bias)
