'''
This is a utilities module for any methods often used by other modules
Things like saving models and parsing board states go here
'''
import torch
import os
import datetime

def save_params(state_dict, model_name):
    """save model to file """
    path = get_filepath(model_name)
    torch.save(state_dict, path)


def load_params(feed_forward, model_name):
    """ load model from file """
    path = get_filepath(model_name)
    feed_forward.load_state_dict(torch.load(path))


def save_checkpoint(state, model_name):
    filepath = get_filepath(model_name, True)
    torch.save(state, filepath)


def load_checkpoint(model):
    filepath = get_filepath(model.name, True)
    if os.path.isfile(filepath):
        print("=> loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        model.start_epoch = checkpoint['epoch']
        model.feed_forward.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filepath, checkpoint['epoch']))
        return True
    else:
        print("=> no checkpoint found at '{}'".format(filepath))
        return False

def initialize_weights(modules, mean=0.0, variance=0.1, bias=0):
    """ intialize network weights as Normal Distribution of given mean and variance """
    for module in modules:
        if hasattr(module, "weight"):
            module.weight.data.normal_(mean, variance)
        if hasattr(module, "bias"):
            module.bias.data.fill_(bias)

def get_filepath(model_name, checkpoint=False):
    if checkpoint:
        filename = '{}_checkpoint.pth.tar'.format(model_name)
    else:
        filename = '{}.pth.tar'.format(model_name)

    filepath = os.path.join('Saves', filename)
    return filepath

