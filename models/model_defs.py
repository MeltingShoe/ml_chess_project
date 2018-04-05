from classes.base_model import generate_class
import models.feed_forward as ff
import models.train as tr
import models.perform_action as pa
import models.board_transform as bt
import torch.nn as nn
import torch.optim as optim


# feed_forward must be passed as an instance
fc_test_params = {
    'name': 'fc_test',
    'ff': ff.ChessFC(),
    'tr': tr.default_train,
    'pa': pa.PA_legal_move_values,
    'bt': bt.noTransform,
    'learning_rate': 0.001,
    'optimizer': optim.Adam,
    'loss_function': nn.L1Loss
}
fc_test = generate_class(fc_test_params)

b0_test_params = {
    'name': 'b0_test',
    'ff': ff.ChessFC(),
    'tr': tr.default_train,
    'pa': pa.PA_legal_move_values,
    'bt': bt.split_by_piece_and_side_with_empty,
    'learning_rate': 0.001,
    'optimizer': optim.Adam,
    'loss_function': nn.L1Loss
}
b0_test = generate_class(b0_test_params)
