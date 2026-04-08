from __future__ import annotations

import numpy as np
import torch


class AlbumentationsTransformAdapter:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        return self.transform(image=image)["image"]


class BatchgeneratorsTransformAdapter:
    """Adapter for single-sample 3D volumes.

    Expects a channel-first volume and returns the transformed image payload.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        result = self.transform(image=image)
        if isinstance(result, dict) and "image" in result:
            image = result["image"]
        else:
            image = result
        if isinstance(image, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(image)).float()
        return image
