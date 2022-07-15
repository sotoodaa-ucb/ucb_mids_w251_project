import cv2
import os
import numpy as np
from typing import Tuple
import torch
from mids_plane_classification.models.lib import MODEL_TYPE_DICT, ModelType
from mids_plane_classification.utils.classes import AIRCRAFT_CLASSES


class Engine:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type

        if model_type not in MODEL_TYPE_DICT:
            raise ValueError(f'Invalid model type provided: {model_type}')

        # Instantiate the proper model and obtain checkpoint.
        self.model = MODEL_TYPE_DICT[model_type]['model']()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint = torch.load(
            os.path.join(os.path.dirname(__file__), MODEL_TYPE_DICT[model_type]['checkpoint_path']),
            map_location=torch.device(self.device)
        )

        # Load from checkpoint.
        self.model.load_state_dict(state_dict=self.checkpoint['state_dict'])

        # Set model to evaluation mode.
        self.model.eval()

    def predict(self, image) -> Tuple[int, str]:
        # Convert to expected size.
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

        # Model expects dimensions (C, H, W).
        if image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))

        # Model expects batches for predictinos, expand to (1, C, H, W).
        image = torch.unsqueeze(torch.Tensor(image), dim=0)

        # Inference.
        raw_prediction = self.model(image)

        # Get the label index and class name.
        prediction_index = torch.argmax(raw_prediction)
        prediction_name = AIRCRAFT_CLASSES[prediction_index]

        return prediction_index, prediction_name
