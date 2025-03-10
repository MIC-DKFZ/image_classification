import torchvision.transforms as T
from .policies.base_transform import BaseTransform

class DummyTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self):
        return T.ToTensor()
