import model_defs
from classes import utils
if __name__ == '__main__':
    # a script to test PA. Works now but I have no idea what's calling _render()
    net = model_defs.fc_test
    run = net()
    a, b = run.play_episode()
    c = utils.process_raw_data(a, b, 0.5)
    run.training_session(c, 10)

