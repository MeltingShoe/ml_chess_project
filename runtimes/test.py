from models import model_defs
from classes import utils
import numpy as np
import time
import torch.multiprocessing as mp

if __name__ == '__main__':
    # a script to test PA. Works now but I have no idea what's calling _render()
    net = model_defs.fc_test
    run = net(resume = True)
    n_epochs = 5
    discount_factor = 0.5
    n_episodes = 5
    run.cuda()


    i = 0
    while(i < 1):
        data = utils.generate_data(run, n_episodes, discount_factor)
        utils.training_session(run, data, n_epochs, 
        	checkpoint_frequency=1, 
        	save_param_frequency=10,
        	print_checkpoint=True)
        i += 1

    


