from classes.base_model import generate_class
import models.feed_forward as ff
import models.train as tr
import models.perform_action as pa
'''
I think all model classes will be defined in this file
We could have a single file for ff, train, and perform action as well
to simplify imports
'''


# feed_forward must be passed as an instance
cifar10_model = generate_class(
    ff.BasicConvNet(), tr.default_train, pa.supervised_evaluate)
