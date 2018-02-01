import torch
import classes.utils as utils
import torch.nn as nn
import torch.optim as optim
from classes.interfaces import BaseModel

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


def generate_class(ff, tr, pa):
    '''
    Setting BaseModel doesn't actually do anything atm because we
    explicitely define the abstract methods in here
    I don't think it really matters though because explicitely defining the
    methods should prevent any inheritance issues
    '''
    superclasses = (BaseModel,)

    '''
    I felt that defining training params inside the model was cleaner and
    made usage easier. I also set defaults for everything,
    we can change these if we want
    '''

    def init(self,
             loss_function=nn.CrossEntropyLoss,
             optimizer=optim.Adam,
             learning_rate=0.001,
             use_cuda=True,
             resume=False,
             filepath='auto'):

        self.use_cuda = use_cuda and torch.cuda.is_available()
        # We might have to add logic in here to set the filepath, not sure
        self.filepath = filepath
        self.trainable_params = ff.parameters()
        # pytorch gets really mad if you don't make the ff it's own class
        self.feed_forward = ff
        self.loss_function = loss_function()
        self.optimizer = optimizer(self.trainable_params, lr=learning_rate)
        if self.use_cuda:
            self.cuda()
            # From what I can tell this is what actually enables cuda
            self.feed_forward.cuda()
        '''
        I'm fairly certain I broke saves/checkpoints
        '''
        if resume:
            # TODO decide a pattern for filenames
            if not utils.load_checkpoint(self):
                self.start_epoch = 0
                utils.initialize_weights(self.feed_forward.modules())
        else:
            self.start_epoch = 0
            utils.initialize_weights(self.feed_forward.modules())

    def training_session(self, train_data, test_data, n_epochs):
        """function to manage the train session"""
        for epoch in range(self.start_epoch, n_epochs, 1):
            self.train(self.feed_forward, train_data, epoch)
            self.evaluate(self.feed_forward, test_data)

        utils.save_params(
            self.feed_forward.state_dict(), self.filepath)
    # not sure if this actually does anything

    def cuda(self):
        self.use_cuda = True

    attrs = {'__init__': init,
             'training_session': training_session,
             'cuda': cuda,
             'train': tr,
             'evaluate': pa
             }
    base_model = type('base_model', superclasses, attrs)
    return base_model
