import os
import cv2
import onnx
import torch
import onnxruntime
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

from mids_plane_classification.models.lib import MODEL_TYPE_DICT, ModelType
from mids_plane_classification.utils.classes import AIRCRAFT_CLASSES


class Engine(ABC):
    """Abstract base class of an inference engine. All possible inference
    engines including PyTorch, ONNX, TensorRT, etc, should extend this
    object. Should contain all common logic related to preprocessing and
    inference for the various inference engines.

    Attributes:
        model_type (ModelType): The type of model (enum).
    """

    def __init__(self, model_type: ModelType):
        if model_type not in MODEL_TYPE_DICT:
            raise ValueError(f'Invalid model type provided: {model_type}')

        self.model_type = model_type
        self.model = MODEL_TYPE_DICT[model_type]['model']()
        self.checkpoint_path = os.path.join(os.path.dirname(__file__), MODEL_TYPE_DICT[model_type]['checkpoint_path'])

    def preprocess(self, image) -> torch.Tensor:
        # Convert to expected size.
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

        # Model expects dimensions (C, H, W).
        if image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))

        # Model expects batches for predictinos, expand to (1, C, H, W).
        return torch.unsqueeze(torch.Tensor(image), dim=0)

    @abstractmethod
    def predict(self, image) -> Tuple[int, str]:
        raise NotImplementedError()


class OnnxEngine(Engine):
    def __init__(self, model_type: ModelType):
        super(OnnxEngine, self).__init__(model_type)
        # Load from the checkpoint, check that model loads correctly.
        self.onnx_model = onnx.load(self.checkpoint_path)
        onnx.checker.check_model(self.onnx_model)

        # Create onnx runtime session.
        self.ort_session = onnxruntime.InferenceSession(self.checkpoint_path)

    def predict(self, image) -> Tuple[int, str]:
        tensor_image = super(OnnxEngine, self).preprocess(image)

        # Compute ONNX Runtime output prediction.
        ort_inputs = {self.ort_session.get_inputs()[0].name: np.array(tensor_image)}
        ort_outs = self.ort_session.run(None, ort_inputs)

        # Get the label index and class name.
        prediction_index = np.argmax(ort_outs)
        prediction_name = AIRCRAFT_CLASSES[prediction_index]

        return prediction_index, prediction_name


class PyTorchEngine(Engine):
    def __init__(self, model_type: ModelType):
        super(PyTorchEngine, self).__init__(model_type)

        # Only PyTorch runtime requires a device specification.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Instantiate the proper model and obtain checkpoint.
        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=torch.device(self.device)
        )

        # Load from checkpoint.
        self.model.load_state_dict(state_dict=self.checkpoint['state_dict'])

        # Set model to evaluation mode.
        self.model.eval()

    def predict(self, image) -> Tuple[int, str]:
        tensor_image = super(PyTorchEngine, self).preprocess(image)

        # Inference.
        raw_prediction = self.model(tensor_image)

        # Get the label index and class name.
        prediction_index = torch.argmax(raw_prediction)
        prediction_name = AIRCRAFT_CLASSES[prediction_index]

        return prediction_index, prediction_name
