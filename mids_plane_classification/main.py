import numpy as np
from PIL import Image
from mids_plane_classification.inference import ModelType, OnnxEngine
from mids_plane_classification.publisher import Publisher
from mids_plane_classification.utils.publisher import API_GATEWAY_URL


def main():
    # Input will likely be a bytes object, we'll need to convert that to a numpy ndarray later.
    img = np.array(Image.open('../res/test_image.png').convert('RGB'))

    # Simple instantiation of the model.
    engine = OnnxEngine(ModelType.ONNX_RESNET18)

    # Instantiate the S3 publisher.
    publisher = Publisher(API_GATEWAY_URL, verbose=True)

    # Inference engine should handle resizing, tensor conversion, etc.
    index, label = engine.predict(img)

    # Publish results to S3.
    publisher.publish(img, label)

    print(index, label)


if __name__ == '__main__':
    main()
