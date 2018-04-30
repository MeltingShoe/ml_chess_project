import sys
import os
import platform
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from classes import utils
from models import model_defs


# add itself for relative imports (TODO: this should be `pip install`(ed)
do_modify = True
# <-- change this if your top level directory is not named ml_chess_project
proj_path = '../ml_chess_project'
for path in sys.path:
    if proj_path == path:
        do_modify = False
        break
if do_modify:
    sys.path.insert(0, proj_path)

net = model_defs.simple_cnn


def pack_episode():

    model = net(resume=True, parent_process=False)
    a, b, metrics = utils.play_episode(
        model, half_turn_limit=model.half_turn_limit)
    split = utils.split_episode_data(a, b)
    white_rewards = utils.discount_reward(
        split['white_rewards'], model.discount_factor)
    black_rewards = utils.discount_reward(
        split['black_rewards'], model.discount_factor)
    states = split['white_states'] + split['black_states']
    rewards = white_rewards + black_rewards
    out = {'states': states, 'rewards': rewards, 'metrics': metrics}
    return out


if __name__ == '__main__':

    if platform.system() == 'Linux':
        mp.set_start_method('spawn')

    utils.ensure_dir(utils.SAVE_DIR)

    def async(n_episodes, n_threads):
        stack = []
        with Pool(processes=n_threads) as pool:
            results = [pool.apply_async(pack_episode)
                       for _ in range(n_episodes)]
            for res in results:
                stack.append(res.get())
            return stack

    def async_generate_data(n_episodes, batch_size,
                            n_threads=os.cpu_count() if os.cpu_count() else 4):
        data = async(n_episodes, n_threads)
        rewards_stack = []
        states_stack = []
        c = 0
        metrics = {'wins': 0, 'moves': 0}
        for _ in range(n_threads):
            c += 1
            a = data.pop()
            rewards_stack += a['rewards']
            states_stack += a['states']
            metrics['wins'] += a['metrics']['wins']
            metrics['moves'] += a['metrics']['moves']

        dataloader = utils.create_dataloader(
            states_stack, rewards_stack, batch_size=batch_size)
        return dataloader, metrics
    run = net(resume=True)
    run.board()

    num_wins = 0
    num_games = 0
    num_moves = 0
    run = net(resume=True)
    for i in range(100000):
        data, metrics = async_generate_data(
            run.episodes_before_update, run.batch_size, n_threads=5)
        # data, metrics = utils.generate_data(run, 5, discount_factor)
        num_moves += metrics['moves']
        num_wins += metrics['wins']
        num_games += 10
        print('Percent of games not drawn:', num_wins / num_games)
        print('Average n_moves:', num_moves / num_games)
        if(i % 100 == 0):
            num_wins = 0
            num_games = 0
            num_moves = 0
        utils.training_session(run, data, run.n_epochs,
                               checkpoint_frequency=1,
                               save_param_frequency=10,
                               starting_index=0,
                               print_checkpoint=False,
                               print_saves=True)
