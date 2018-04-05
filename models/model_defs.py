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

fc_12_slice_params = {
    'name': 'fc_12_slice',
    'ff': ff.FC12Slice(),
    'tr': tr.default_train,
    'pa': pa.PA_legal_move_values,
    'bt': bt.split_by_piece_and_side,
    'learning_rate': 0.001,
    'optimizer': optim.Adam,
    'loss_function': nn.L1Loss
}
fc_12_slice = generate_class(fc_12_slice_params)
