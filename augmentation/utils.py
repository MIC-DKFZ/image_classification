import torchvision.transforms as T


class DummyTransform:
    def __call__(self):
        return T.ToTensor()
