import numpy as np
from PIL import Image
from mids_plane_classification.inference import Engine, ModelType


def main():
    # Input will likely be a bytes object, we'll need to convert that to a numpy ndarray later.
    img = np.array(Image.open('../res/test_image.png').convert('RGB'))

    # Simple instantiation of the model.
    engine = Engine(ModelType.RESNET18)

    # Inference engine should handle resizing, tensor conversion, etc.
    index, label = engine.predict(img)

    print(index, label)


if __name__ == '__main__':
    main()
