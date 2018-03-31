import torch
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

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


def discount_reward(rewards_list, discount_factor):
    '''
    might not be the most efficient
    rewards_list must be a list with only the rewards
    training data must be split seperately
    '''
    i = 0
    while i < len(rewards_list):
        j = i
        exp = 1
        while j > 1:
            _ = rewards_list[i]
            rewards_list[j - 1] += _ * discount_factor**exp
            j -= 1
            exp += 1
        i += 1
    return rewards_list

#splits up black and whites training data
def split_episode_data(states, rewards):
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
    out = {
    'white_states': white_states,
    'white_rewards': white_rewards,
    'black_states': black_states,
    'black_rewards': black_rewards
    }
    return out

def create_dataloader(states, rewards):
    states = torch.Tensor(np.array(states).tolist())
    rewards = torch.Tensor(rewards).unsqueeze(1)
    dataset = data_utils.TensorDataset(states, rewards)
    dataloader = data_utils.DataLoader(dataset)
    return dataloader

#encapsulates split_episode_data, discount_reward, and create_dataloader    
def process_raw_data(states, rewards, discount_factor, cat=True):
    split = split_episode_data(states, rewards)
    white_rewards = discount_reward(split['white_rewards'], discount_factor)
    black_rewards = discount_reward(split['black_rewards'], discount_factor)
    if cat:
        states = split['white_states']+split['black_states']
        rewards = white_rewards + black_rewards
        dataloader = create_dataloader(states, rewards)
        return dataloader
    split_data = {
        'white_data': create_dataloader(split['white_states'], white_rewards),
        'black_data': create_dataloader(split['black_states'], black_rewards)
    }
    return split_data
