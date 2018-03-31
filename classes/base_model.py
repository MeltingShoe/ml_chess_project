import inspect
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from classes.interfaces import BaseModel
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
    explicitely define the abstract methods in here
    I don't think it really matters though because explicitely defining the
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
        return None

    def init(self,
             use_cuda=True,
             resume=False):
        ''''
        Most of these attributes should be moved to the attrs dict
        It would help readability and avoid some inheritence
        collisions if that ever comes up
        '''
        self.use_cuda = use_cuda and torch.cuda.is_available()
        # We might have to add logic in here to set the filepath, not sure
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

    def training_session(self, dataset, n_epochs):
        """function to manage the train session"""
        for epoch in range(self.start_epoch, n_epochs, 1):
            self.train(self.feed_forward, dataset, epoch)

        utils.save_params(
            self.feed_forward.state_dict(), self.name)

    def play_episode(self):
        self.env._reset()
        states = []
        rewards = []
        while True:
            # going to have to add more complex logic for policy estimation networks
            a = self.perform_action()
            states.append(a['state'])
            rewards.append(a['reward'])
            print(a['isTerminated'])
            # not taking the effort to come up with a proper struct because that's planned for later
            if(a['isTerminated'] == True):
                return states, rewards

    def calc_future_reward(self, states, rewards, discount_factor, split_sides=False):
        # split states/rewards into their two sides
        i = 0
        white_states = []
        white_rewards = []
        black_states = []
        black_rewards = []
        while(i < len(states)):
            if(i % 2 == 0):
                white_states.append(states[i])
                white_rewards.append(rewards[i])
            else:
                black_states.append(states[i])
                black_rewards.append(rewards[i])
            i += 1
        # calc future reward
        white_rewards = torch.Tensor(utils.discount_reward(white_rewards, discount_factor))
        black_rewards = torch.Tensor(utils.discount_reward(black_rewards, discount_factor))
        white_rewards = white_rewards.unsqueeze(1)
        black_rewards = black_rewards.unsqueeze(1)
        white_states = torch.Tensor(np.array(white_states).tolist())
        black_states = torch.Tensor(np.array(black_states).tolist())

        if split_sides:
            white_dataset = data_utils.TensorDataset(white_states, white_rewards)
            black_dataset = data_utils.TensorDataset(black_states, black_rewards)
            white_dataloader = data_utils.DataLoader(white_dataset)
            black_dataloader = data_utils.DataLoader(black_dataset)
            training_data = {
                'white_dataloader': white_dataloader,
                'black_dataloader': black_dataloader
            }
        else:
            states = torch.cat((white_states, black_states), 0)
            rewards = torch.cat((white_rewards, black_rewards), 0)
            dataset = data_utils.TensorDataset(states, rewards)
            dataloader = data_utils.DataLoader(dataset)
            training_data = {'dataloader': dataloader}

        return training_data

    # not sure if this actually does anything

    def cuda(self):
        self.use_cuda = True

    # we should check if params are rendundant (e.g. optimizer and learning rate) or if we need other params
    attrs = {'__init__': init,
             'training_session': training_session,
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
             'play_episode': play_episode,
             'calc_future_reward': calc_future_reward,
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
