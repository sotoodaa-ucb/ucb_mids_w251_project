import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet


class AircraftResNet18(ResNet):
    """
    Extend the default Reset18 model with a
    slight modification to the output linear layer.
    """
    def __init__(self, num_classes=11):
        super(AircraftResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.fc = nn.Linear(self.fc.in_features, num_classes)
