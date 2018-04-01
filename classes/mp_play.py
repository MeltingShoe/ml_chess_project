from classes import utils

class MultiPlay():

    def __init__(self, model, discount_factor):
        self.net = model(resume = True)
        self.discount_factor = discount_factor
        self.stack = []


    def pack_episode(self):
        print('paq')
        a, b = utils.play_episode(self.net)
        split = utils.split_episode_data(a, b)
        white_rewards = utils.discount_reward(split['white_rewards'], self.discount_factor)
        black_rewards = utils.discount_reward(split['black_rewards'], self.discount_factor)
        states = split['white_states']+split['black_states']
        rewards = white_rewards + black_rewards
        out = {'states': states, 'rewards': rewards}
        self.stack.append(out)

    def get_dataloader(self):
        state_stack = []
        reward_stack = []
        for item in self.stack:
            a = self.stack.pop()
            state_stack += (a['states'])
            reward_stack += (a['rewards'])
        data = utils.create_dataloader(state_stack, reward_stack)
        self.stack = []
        return data

