import torch.nn as nn
import torch.nn.functional as F
from .models.resnet import ResNet
from .models.mobilenet import MobileNetV2


__all__ = ['Net','SimpleNet', 'ResNet', 'MobileNetV2']


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class SimpleNet(nn.Module):

    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super(SimpleNet, self).__init__()
        h, w, ch = input_shape
        # kernels
        self.conv1 = nn.Conv2d(ch, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * int((h-4)/4) * int((w-8)/4), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)             # 5: number of classes  (without background)
        # self.dout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, num_flat_features(x))
        # x = F.relu(self.fc1(x))
        x = F.relu(nn.Linear(num_flat_features(x), 120)(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





def Net(model_name, **kwargs):

    if model_name not in __all__:
        print('No model named %s' %model_name)
        return None

    return eval(model_name)(**kwargs)
