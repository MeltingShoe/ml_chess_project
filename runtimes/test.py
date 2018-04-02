import sys
# add itself for relative imports (TODO: this should be `pip install`(ed) but the project itself may need a few `__init__` definitions to make it into a python module)
do_modify = True
# <-- change this if your top level directory is not named ml_chess_project
proj_path = '../ml_chess_project'
for path in sys.path:
    if proj_path == path:
        do_modify = False
        break
if do_modify:
    sys.path.insert(0, proj_path)

from torch.multiprocessing import Pool

from classes import utils
from models import model_defs


net = model_defs.fc_test
n_epochs = 20
discount_factor = 0.5


def pack_episode():

    a, b, metrics = utils.play_episode(net(resume=True, parent_process=False))
    split = utils.split_episode_data(a, b)
    white_rewards = utils.discount_reward(
        split['white_rewards'], discount_factor)
    black_rewards = utils.discount_reward(
        split['black_rewards'], discount_factor)
    states = split['white_states'] + split['black_states']
    rewards = white_rewards + black_rewards
    out = {'states': states, 'rewards': rewards, 'metrics': metrics}
    return out


if __name__ == '__main__':

    utils.ensure_dir(utils.SAVE_DIR)

    def async(threads):
        stack = []
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
        metrics = {'wins': 0}
        for _ in range(5):
            c += 1
            a = data.pop()
            rewards_stack += a['rewards']
            states_stack += a['states']
            metrics['wins'] += a['metrics']['wins']

        dataloader = utils.create_dataloader(states_stack, rewards_stack)
        return dataloader, metrics

    num_wins = 0
    num_games = 0
    run = net(resume=True)
    for i in range(3):
        data, metrics = async_generate_data()
        #data, metrics = utils.generate_data(run, 5, discount_factor)
        num_wins += metrics['wins']
        num_games += 5
        print('Percent of games not drawn:', num_wins / num_games)
        utils.training_session(run, data, n_epochs,
                               checkpoint_frequency=1,
                               save_param_frequency=10,
                               starting_index=0,
                               print_checkpoint=True,
                               print_saves=True)
