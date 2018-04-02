# ml_chess_project

## Pre-requisites:
Below are a list of packages that are required to get the game to work, along with links for installation instructions.

### python-chess
For usage and installation instructions, see: https://pypi.python.org/pypi/python-chess

### OpenAI Gym
For installation instructions, see: https://www.youtube.com/watch?v=Io1wOuHEyW0&feature=youtu.be

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

## load_checkpoint(model, print_out=True)
Loads a checkpoint for a model. Currently used in the model class init if a checkpoint for that model is present.

Parameters:
```
model: The base container class of the model to be saved
print_out=True: When True this will print a message to the console
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

# /classes/base_model.py
This file contains generate_class, which is a "class factory" function that generates a class encapsulating all the functionality required for a model to work.
The class uses typeclass style construction, but this isn't functionally any different from normal class construction. It just made writing this function easier.
The name base_model is inaccurate, because this class is made as a container for all of the parameters of a single experiment such as the training function, learning rate, and much more.
The class also contains an instance of the gym environment where games are played.

## generate_class parameters
The parameters of this function are expected to be a dict containing specific keys.

Parameters:
```
'name': The name of the model/experiment to be used in the filename of checkpoints and parameter saves
'ff': A class containing the feed forward method of the model defined in /models/feed_forward.py
'tr': A function for training the model defined in /models/train.py. The first param of this function should be self
'pa': A function that should feed the current board position through feed_forward and play a move. 
'bt': A function that takes the board position and makes any necessary transformations, such as encoding the position as an 8x8x12 array or autoencoding the position
'learning_rate': The learning rate to use
'optimizer': The optimizer to use
'loss_function': The loss function to use
```

## init(self, parent_process=True, use_cuda=True, resume=True)
The classes init which sets whether cuda is used and loads a checkpoint if one exists, or initialized weights if it doesn't.

Parameters:
```
use_cuda=True: Whether cuda should be used.
resume=True: If this is false new weights will be initialized no matter what. Be careful with this because the models old parameters will be overwritten.
parent_process=True: This is set to false for async processes and determines whether to print the loading_checkpoint message. Unfortunately the async processes call init twice with the first time using the same params as main process so it still prints out the message, but at least it only prints 5 instead of 10 now. I have no idea why this is the case.
```

## cuda(self)
Sets self.use_cuda to True

## check_params(params)
Takes in the params dict and does some checks to make sure the right keys are given and types are correct.
Needs to be updated to also check for board_transform

# /models/board_transform.py
This file is for funtions that modify the board representation, like splitting it into an 8x8x12 array or autoencoding the board state. 
These functions should take self as a parameter and modify `self.env._get_array_state()[0]`

## noTransform(self)
This function does not modify the board representation. It's still necessary because wherever something trys to get the board state it calls `self.board()`

# /models/feed_forward.py
This file is for defining the feed forward computation of all models we build. These should be classes that subclass `nn.module` and have a method named `forward()` that takes a single
parameter as the input for the network

## ChessFC()
This is a fully connected network that was build to test functionality of the program but for some reason is performing far better than expected.

It's structure is:
```
64 input nodes
FC 128 nodes
FC 256 nodes
Dropout p=0.5
FC 512 nodes
Dropout p=0.5
FC 512 nodes
Dropout p=0.5
FC 256 nodes
Dropout p=0.5
FC 64 nodes
FC 32 nodes
1 output node
```
Relu is applied to all but the last layer. All decisions about the structure of this network were completely arbitrary. It was only built as a test and no thought was put into
making a network that would perform well, this was only meant to test that our code works.

# /models/model_defs.py
This file is for defining every model we build. To build a new experiment write whatever code you need in board_transform.py, feed_forward.py, perform_action.py, and train.py. 
Then in this file make a new dict defining all the params, and under it write `model_name = generate_class(params_dict)`
For params documentation see the documentation for /classes/base_model.py

# /models/perform_action.py
This file is for functions that call feed_forward on the board state, select/play a move, and should output a dict containing the state, reward, isTerminated, and the chosen move.

## PA_legal_move_values()
This has the network evaluate the position after every possible legal move, softmaxes the outputs, and selects a move from the resulting probability distribution. This requires
that the network only outputs a single value that represents the value of a given position.

# /models/train.py
This file is for training functions.

## default_train(self, dataloader, starting_index=0, print_batch=False)
A training function that should work for most networks we make.

Parameters:
```
dataloader: a pytorch dataloader of the training data
starting_index=0: the index to start training from, in case you want to halt training and start again later at that index
print_batch=False: if true the epoch loss will be printed every batch
```

# /runtimes/test.py
Our main runtime, named test because I made this file while debugging and planned on making a new runtime. 

```
net = model_defs.model: the experiment to be run defined in /models/model_defs.py
n_epochs: The number of epochs to train on each set of generated data
discount_factor: An int for how much importance should be given to future rewards
```

## pack_episode()
This is a function called by async() that calls utils.play_episode(), utils.split_episode_data(), utils.discount_reward(), concatenates the output, and returns it.

## async()
This uses torch.multiprocessing to spawn 5 processes to play 5 games and return their data in a list. The number of games is currently not dynamic because I couldn't figure out how to do that.
Right now multiprocessing is fairly buggy and can cause python to crash if it gets a keyboard interrupt. As such it should mainly be used for training networks because it's around
twice as fast as utils.generate_data(), but utils.generate_data() can be used for debugging so you don't have to restart your interpreter every time you have to force the process to quit.

## async_generate_data()
This calls async, concatenates the resulting data, and packs it all into a dataloader.

## Runtime loop
The runtime loop is simply a for loop that calls either async_generate_data() or utils.generate_data() followed by utils.training_session()
