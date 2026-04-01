import numpy as np


class AlbumentationsTransformAdapter:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        return self.transform(image=image)["image"]
