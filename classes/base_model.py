import inspect
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import classes.utils as utils
import gym
import gym_chess
import numpy as np


'''
This is "class factory" function that generates a new class for each model
It uses typeclass construction primarily for convinience, but essentially
works the same way as normal class construction

Params
ff: feed_forward. Must be a class for autograd/cuda to work
tr: The training function. This is a function, not a class.
    Params are instantiated in here
pa: Currently the supervised evaluate function, we'll need
    to change some things for perform_action
'''


def generate_class(params):
    '''
    Setting BaseModel doesn't actually do anything atm because we
    explicitly define the abstract methods in here
    I don't think it really matters though because explicitly defining the
    methods should prevent any inheritance issues
    Update: no longer uses ABC because it caused problems
    '''
    superclasses = (object,)

    '''
    I felt that defining training params inside the model was cleaner and
    made usage easier. I also set defaults for everything,
    we can change these if we want
    '''
    # check if all the params are correct
    if not check_params(params):
        print('check_params failed')
        return None

    def init(self, parent_process=True, use_cuda=True, resume=True):

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
            # From what I can tell this is what actually enables cuda
            self.feed_forward.cuda()

        if resume:
            # TODO decide a pattern for filenames

            if not utils.load_checkpoint(self, print_out=parent_process):
                self.epoch = 0
                utils.initialize_weights(self.feed_forward.modules())
        else:
            self.epoch = 0
            utils.initialize_weights(self.feed_forward.modules())

    # not sure if this actually does anything

    def cuda(self):
        self.use_cuda = True

    # we should check if params are rendundant (e.g. optimizer and learning rate) or if we need other params
    attrs = {'__init__': init,
             'cuda': cuda,
             'train': params['tr'],
             'perform_action': params['pa'],
             'feed_forward': params['ff'],
             'trainable_params': params['ff'].parameters(),
             'name': params['name'],
             'learning_rate': params['learning_rate'],
             'optimizer': params['optimizer'](params['ff'].parameters(), lr=params['learning_rate']),
             'loss_function': params['loss_function'](),
             'env': gym.make('chess-v0'),
             'board': params['bt']
             }
    base_model = type('base_model', superclasses, attrs)
    return base_model


def check_params(params):
    # check if keys exists in the dictionary
    first_check = False
    type_check = False
    keys_check = set(['learning_rate', 'loss_function', 'name',
                      'optimizer', 'ff', 'tr', 'pa']).issubset(params)
    if keys_check:
        lr = params['learning_rate']
        lf = params['loss_function']
        name = params['name']
        opt = params['optimizer']
        ff = params['ff']
        tr = params['tr']
        pa = params['pa']

        # check on ff, tr, pa
        first_check = issubclass(ff.__class__, nn.Module) and inspect.isfunction(
            tr) and inspect.isfunction(pa)
        # not completely sure if the last two 'has_attr' work as i think (confirmed that it doesn't)
        type_check = isinstance(lr, float) and lr < 1 and isinstance(
            name, str)  # and hasattr(nn, lf) and hasattr(optim, opt)

    return first_check and type_check
