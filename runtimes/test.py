from models import model_defs
from classes import utils
from classes import mp_play
import numpy as np
import time
import torch.multiprocessing as mp
#from torch.multiprocessing import Queue

net = model_defs.fc_test
stack = []
n_epochs = 5
discount_factor = 0.5
n_episodes = 5
def pack_episode():
    print('paq')
    a, b = utils.play_episode(net(resume=True))
    split = utils.split_episode_data(a, b)
    white_rewards = utils.discount_reward(split['white_rewards'], discount_factor)
    black_rewards = utils.discount_reward(split['black_rewards'], discount_factor)
    states = split['white_states']+split['black_states']
    rewards = white_rewards + black_rewards
    out = {'states': states, 'rewards': rewards}
    stack.append(out)


def get_dataloader(stack):
    state_stack = []
    reward_stack = []
    for item in stack:
        a = stack.pop()
        state_stack += (a['states'])
        reward_stack += (a['rewards'])
    data = utils.create_dataloader(state_stack, reward_stack)
    return data



if __name__ == '__main__':
    # a script to test PA. Works now but I have no idea what's calling _render()
    run = net(resume = True)
    run.cuda()

    for i in range(3):
        print('ooo')
        p = mp.Process(target=pack_episode)
        p.start()
    #p.join()
    print('unjoin')
    theD = get_dataloader(stack)
    print(theD)






