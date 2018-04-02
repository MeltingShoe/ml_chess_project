import torch
import os
import numpy as np
import torch.utils.data as data_utils

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

SAVE_DIR = "Saves"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_params(model, print_out=True):
    """save model to file """
    if print_out:
        print('Saving params. Epoch:', model.epoch)
    path = get_filepath(model.name)
    torch.save(model.feed_forward.state_dict(), path)


def load_params(model):
    """ load model from file """
    path = get_filepath(model.name)
    model.feed_forward.load_state_dict(torch.load(path))


def save_checkpoint(model, print_out=True):
    if print_out:
        print('Saving checkpoint. Epoch:', model.epoch)
    filepath = get_filepath(model.name, True)
    state = {
        'epoch': model.epoch,
        'state_dict': model.feed_forward.state_dict(),
        'optimizer': model.optimizer.state_dict()}
    torch.save(state, filepath)


def load_checkpoint(model):
    filepath = get_filepath(model.name, True)
    if os.path.isfile(filepath):
        print("=> loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        model.epoch = checkpoint['epoch']
        model.feed_forward.load_state_dict(checkpoint['state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filepath, checkpoint['epoch']))
        return True
    else:
        print("=> no checkpoint found at '{}'".format(filepath))
        return False


def initialize_weights(modules, mean=0.0, variance=0.1, bias=0):
    """ initialize network weights as Normal Distribution"""
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

    filepath = os.path.join(SAVE_DIR, filename)
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

# splits up black and whites training data


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


def create_dataloader(states, rewards, batch_size=32):
    states = torch.Tensor(np.array(states).tolist())
    rewards = torch.Tensor(rewards).unsqueeze(1)
    dataset = data_utils.TensorDataset(states, rewards)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size)
    return dataloader


def process_raw_data(states, rewards, discount_factor, cat=True):
    split = split_episode_data(states, rewards)
    white_rewards = discount_reward(split['white_rewards'], discount_factor)
    black_rewards = discount_reward(split['black_rewards'], discount_factor)
    if cat:
        states = split['white_states'] + split['black_states']
        rewards = white_rewards + black_rewards
        dataloader = create_dataloader(states, rewards)
        return dataloader
    split_data = {
        'white_data': create_dataloader(split['white_states'], white_rewards),
        'black_data': create_dataloader(split['black_states'], black_rewards)
    }
    return split_data


def training_session(model, dataloader, n_epochs,
                     checkpoint_frequency=1, save_param_frequency=10,
                     starting_index=0, print_batch=False,
                     print_checkpoint=True, print_saves=True):
    """function to manage the train session"""
    for epoch in range(0, n_epochs, 1):
        model.train(dataloader,
                    starting_index=starting_index,
                    print_batch=print_batch)
        if(checkpoint_frequency is not None):
            if(model.epoch % checkpoint_frequency == 0):
                save_checkpoint(model, print_out=print_checkpoint)
        if(save_param_frequency is not None):
            if(model.epoch % save_param_frequency == 0):
                save_params(model, print_out=print_saves)


def play_episode(model, half_turn_limit=2000, print_rewards=True):
    model.env._reset()
    states = []
    rewards = []
    i = 0
    while True:
        a = model.perform_action()
        states.append(a['state'])
        rewards.append(a['reward'])
        if a['isTerminated'] or i > half_turn_limit:
            if print_rewards:
                print(sum(rewards), len(rewards))
            return states, rewards
        i += 1

# This plays multiple episodes and packs them all in 1 dataloader. Improves speed by ~8%


def generate_data(model, num_games, discount_factor,
                  half_turn_limit=2000, print_rewards=True):
    i = 0
    states = []
    rewards = []
    while(i < num_games):
        raw_state, raw_reward = play_episode(model,
                                             half_turn_limit=half_turn_limit, print_rewards=print_rewards)
        split = split_episode_data(raw_state, raw_reward)
        white_rewards = discount_reward(
            split['white_rewards'], discount_factor)
        black_rewards = discount_reward(
            split['black_rewards'], discount_factor)
        states += split['white_states']
        states += split['black_states']
        rewards += white_rewards
        rewards += black_rewards
        i += 1
    dataloader = create_dataloader(states, rewards)
    return dataloader
