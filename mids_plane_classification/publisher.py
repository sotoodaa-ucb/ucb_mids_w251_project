import cv2
import requests
import datetime
import numpy as np
from typing import Tuple


class Publisher:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {'Content-Type': 'image/jpeg'}

    def publish(self, image: np.array, classification: str, verbose: bool = False) -> Tuple[int, str]:
        # Get the current timestamp to mark inference time.
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')

        # Encode image to png and convert to bytes.
        _, encoded_image = cv2.imencode('.png', image)
        img = encoded_image.tobytes()

        # Build the API Gateway URL, concatenating the timestamp and classification as the file name.
        url = f'{self.base_url}/{timestamp}_{classification}.png'

        # Submit the request.
        response = requests.post(url, headers=self.headers, data=img)

        if verbose:
            print(url, response.status_code)

        return response.status_code, url
