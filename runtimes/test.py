from models import model_defs
from classes import utils
import numpy as np

if __name__ == '__main__':
    # a script to test PA. Works now but I have no idea what's calling _render()
    net = model_defs.fc_test
    run = net(resume = True)
    run.cuda()
    i = 0
    length = 0
    while(i < 10000):
        a, b = run.play_episode()
        if(len(b) == length):
        	run.learning_rate += 0.001
        	print(run.learning_rate)
        else:
        	run.learning_rate == 0.001
        length = len(b)
        c = utils.process_raw_data(a, b, 0.5)
        run.training_session(c, 5)
        run.start_epoch = 0
        i += 1
