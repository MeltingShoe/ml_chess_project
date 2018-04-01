from models import model_defs
from classes import utils
import numpy as np
import time
import torch.multiprocessing as mp
#from torch.multiprocessing import pool




def pack_episode(model, discount_factor):
    a, b = utils.play_episode(model)
    split = utils.split_episode_data(a, b)
    white_rewards = utils.discount_reward(split['white_rewards'], discount_factor)
    black_rewards = utils.discount_reward(split['black_rewards'], discount_factor)
    states = split['white_states']+split['black_states']
    rewards = white_rewards + black_rewards
    return states, rewards

def wrap():
	net = model_defs.fc_test
	run = net(resume = True)
	run.cuda()
	a,b = pack_episode(run, 0.5)
	print('kkk')
	return a,b

if __name__ == '__main__':
	
    # a script to test PA. Works now but I have no idea what's calling _render()
    net = model_defs.fc_test
    run = net(resume = True)
    run.cuda()
    n_epochs = 5
    discount_factor = 0.5
    n_episodes = 5
    
    '''
    start = time.time()
    q = mp.Queue
    for i in range(5):
        print('ooo')
        p = mp.Process(target=wrap)
        p.start()
    p.join()
    print('multiprocessing', time.time()-start)
    '''
    start= time.time()

    i = 0
    while(i < 1):
        data = utils.generate_data(run, n_episodes, discount_factor)
        '''
        utils.training_session(run, data, n_epochs, 
        	checkpoint_frequency=1, 
        	save_param_frequency=10,
        	print_checkpoint=True)'''
       
        i += 1
    print('the gayer way', time.time()-start)
