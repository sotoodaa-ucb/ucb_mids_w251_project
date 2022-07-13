from enum import Enum
from mids_plane_classification.models.resnet18 import AircraftResNet18


class ModelType(Enum):
    RESNET18 = 'resnet18'


MODEL_TYPE_DICT = {
    ModelType.RESNET18: {
        'model': AircraftResNet18,
        'checkpoint_path': './models/checkpoints/model_best.pth.tar'
    }
}
