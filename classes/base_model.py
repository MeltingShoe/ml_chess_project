import torch
import classes.utils as utils
import torch.nn as nn
import torch.optim as optim
from classes.interfaces import BaseTrain


def generate_class(ff, tr, pa):
    superclasses = (object,)

    def init(self, feed_forward, loss_function=nn.CrossEntropyLoss, optimizer=optim.Adam, learning_rate=0.001, use_cuda=True, resume=False, filepath='auto'):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.filepath = filepath
        self.trainable_params = feed_forward.parameters()
        self.feed_forward = feed_forward
        # train args
        self.loss_function = loss_function()
        self.optimizer = optimizer(self.trainable_params, lr=learning_rate)
        # from what I've seen this just sets self.use_cuda to true so I'm not sure if this does anything
        if self.use_cuda:
            self.cuda()
            self.feed_forward.cuda()
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
            self.evaluate(self.feed_forward.forward, test_data)

        utils.save_params(
            self.feed_forward.state_dict(), self.filepath)
    def cuda(self):
        self.use_cuda = True
        """
        params:
            feed_forward: The feed forward method of a model
            training_method: The training loop of a model
            evaluate: Method for running a models feed_forward and parsing the output to UCI notation
            use_cuda: a boolean whether to use CUDA, only works if CUDA is installed
            resume: flag to say if we are resuming training
            filepath: file where to save the model
        """
    attrs = {'__init__': init, 'training_session': training_session, 'cuda': cuda,
             'train': tr, 'evaluate': pa
                }
    base_model = type('base_model', superclasses, attrs)
    return base_model
