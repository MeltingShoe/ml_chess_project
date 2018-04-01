from models import model_defs
from classes import utils
import numpy as np
import time
import torch.multiprocessing as mp
from torch.multiprocessing import Pool

net = model_defs.fc_test
n_epochs = 20
discount_factor = 0.5
run = net(resume = True)
run.cuda()


def pack_episode():
    a, b = utils.play_episode(net(resume=True))
    split = utils.split_episode_data(a, b)
    white_rewards = utils.discount_reward(split['white_rewards'], discount_factor)
    black_rewards = utils.discount_reward(split['black_rewards'], discount_factor)
    states = split['white_states']+split['black_states']
    rewards = white_rewards + black_rewards
    out = {'states': states, 'rewards': rewards}
    return out

if __name__ == '__main__':

    utils.ensure_dir(utils.SAVE_DIR)

    def async(threads):
        stack = []
        '''
        I have no idea what am doing and no matter what I tried I couldn't find a way to make the 
        number of total episodes dynamic. Multiprocessing is weird and difficult to work with.
        Also this has a tendency to completely crash python if you do a keyboard interrupt sooooo...
        It's interesting to note that it calls load_checkpoint twice for each process too
        '''
        with Pool(processes=threads) as pool:
            res = pool.apply_async(pack_episode)
            res1 = pool.apply_async(pack_episode)
            res2 = pool.apply_async(pack_episode)
            res3 = pool.apply_async(pack_episode)
            res4 = pool.apply_async(pack_episode)
            stack.append(res.get())
            stack.append(res1.get())
            stack.append(res2.get())
            stack.append(res3.get())
            stack.append(res4.get())
        return stack

    def async_generate_data():
        data = async(5)
        rewards_stack = []
        states_stack = []
        c = 0
        for _ in range(5):
        	c += 1
        	a = data.pop()
        	rewards_stack += a['rewards']
        	states_stack += a['states']

        dataloader = utils.create_dataloader(states_stack, rewards_stack)
        return dataloader


    for i in range(10):
    	data = async_generate_data()
    	#data = utils.generate_data(run, 5, discount_factor)
    	utils.training_session(run, data, n_epochs,
    							checkpoint_frequency=1, 
    							save_param_frequency=10, 
    							starting_index=0,
    							print_checkpoint=True, 
    							print_saves=True)




    





