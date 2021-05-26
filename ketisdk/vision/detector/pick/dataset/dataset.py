from .grip_cifar10_dber import GripCifar10
from .suction_cifar10_dber import SuctionCifar10
from .grasp_cifar10_dber import GraspCifar10
from torchvision.datasets import *
import torchvision


__all__ = ('GripCifar10','SuctionCifar10', 'GraspCifar10') + torchvision.datasets.__all__

def get_mean_std(name):
    assert name in __all__ + torchvision.datasets.__all__
    if name in torchvision.datasets.__all__:
        db_mean = (0.4914, 0.4822, 0.4465)
        db_std = (0.2023, 0.1994, 0.2010)
    else:
        db_mean = eval(name + '.db_mean')
        db_std = eval(name + '.db_std')
    return db_mean, db_std

def Dataset(name, **kwargs):
    assert name in __all__ + torchvision.datasets.__all__
    if name in torchvision.datasets.__all__: del kwargs['im_shape']
    return eval(name)(**kwargs)