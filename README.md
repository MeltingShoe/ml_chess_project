# ml_chess_project

## Pre-requisites:
Below are a list of packages that are required to get the game to work, along with links for installation instructions.

### python-chess
For usage and installation instructions, see: https://pypi.python.org/pypi/python-chess

### OpenAI Gym
For installation instructions, see: https://www.youtube.com/watch?v=Io1wOuHEyW0&feature=youtu.be

## gym-chess has moved!
You can find it here: https://github.com/MeltingShoe/gym-chess

Documentation
======
The most important objective in writing these docs is explaining what everything does, why they were made, and how everything works together. As such I may be brief as to exactly how some parts of the program work, but I will be documenting everything in the project. If the usage of anything is unclear, let me know and I'll update the docs on it.

# /classes/utils.py
The purpose of utils.py is to compile all functions that may be used in more than one place that are smaller and don't have a specific reason to be anywhere else.
No functions in this file should reference self with the intention to be used in another class. Anything that needs to call class methods should instead take that classes instance as input.

## ensure_dir(path)
This should be called before any functions that save/load from a different directory.
Right now it's just used at the start of the runtime to ensure the saves folder exists.

## save_params(model, print_out=True)
This saves the feed_forward parameters of a model.

Parameters:
```
model: The base container class of the model to be saved
print_out=True: When True this will print a message to the console with the models current epoch
```

## load_params(feed_forward, model_name)
This loads the parameters of a model. To be consistent it should probably take the model as input rather than taking it's FF and name as input. 
As far as I know this function is currently not used anywhere.

## save_checkpoint(model, print_out=True)
This saves a models feed_forward parameters, as well as it's optimizer parameters and current epoch.
In the future I'd like to extend this to also save all hyperparameters of the model and possibly the code needed to run the model itself.

Parameters:
```
model: The base container class of the model to be saved
print_out=True: When True this will print a message to the console with the models current epoch
```

## load_checkpoint(model)
Loads a checkpoint for a model. Currently used in the model class init if a checkpoint for that model is present.

Parameters:
```
model: The base container class of the model to be saved
```

## initialize_weights(modules, mean=0.0, variance=0.1, bias=0)
Initializes weights for a new network as normal distribution. 

Parameters:
```
modules: The networks feed_forward modules.
```

## get_filepath(model_name, checkpoint=False)
Gets the filepath to save a model to. 

## discount_reward(rewards_list, discount_factor)
Calculates new reward values with discounted future rewards.

Parameters:
```
rewards_list: A list of rewards from an episode
discount_factor: An int for how much importance should be given to future rewards
```

## split_episode_data(states, rewards)
Splits raw training data from a single episode and returns a dict with training data for black and white.

Parameters:
```
states: a list containing all states of the episode
rewards: a list containing all rewards of the episode
```

## create_dataloader(states, rewards, batch_size=32)
Packs training data into a pytorch dataloader.

Parameters:
```
states: a list containing all states of the episode
rewards: a list containing all rewards of the episode
batch_size: the batch size
```

## process_raw_data(states, rewards, discount_factor, cat=True)
Wraps split_episode_data, discount_reward, and create_dataloader.

Parameters:
```
states: a list containing all states of the episode
rewards: a list containing all rewards of the episode
discount_factor: An int for how much importance should be given to future rewards
cat: if False a dict will be returned with seperate dataloaders for black and white
```

## training_session(model, dataset, n_epochs):
Trains a model for a number of epochs.

Parameters:
```
model: The base container class of the model
dataloader: A dataloader of training data
n_epochs: The number of epochs to train for
checkpoint_frequency=1: Number of epochs that should pass before saving a checkpoint
save_param_frequency=10: Number of epochs that should pass before saving params
starting_index=0: I believe this is to specify if training should start later in the dataset to pick up at a later point
print_batch=False: Whether to printout batch data
print_checkpoint=True: Whether to printout checkpoint saves
print_saves=True: Whether to printout param saves
```

## play_episode(model, half_turn_limit=2000, print_rewards=True)
Plays a single game of chess with the model against itself and outputs raw training data

Parameters:
```
model: The base container class of the model to be saved
half_turn_limit=2000: The number of individual moves before this will terminate without waiting for the game to end
print_rewards=True: If true this will print out the total of all rewards and the number of moves made
```

## generate_data(model, num_games, discount_factor)
Plays multiple games with the model against itself, processes all the data, and packages it into a single dataloader.

Parameters:
```
model: The base container class of the model to be saved
num_games: The number of games to play
discount_factor: An int for how much importance should be given to future rewards
half_turn_limit=2000: The number of individual moves before this will terminate without waiting for the game to end
print_rewards=True: If true this will print out the total of all rewards and the number of moves made
```
