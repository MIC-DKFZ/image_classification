from torch.utils.data import Dataset
from .base_datamodule import BaseDataModule
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PosterRegressionData(Dataset):

    def __init__(self, root, split, transform, albumentation=False):
        super().__init__()
        """
        Images can be downloaded from https://www.kaggle.com/datasets/phiitm/movie-posters
        """

        self.transform = transform
        self.albumentation = albumentation       

        file_list = list(Path(root).glob('*.jpg'))
        files, targets = zip(*[(i, str(i).split('/')[-1].split('_')[0]) for i in file_list])
        X_train, X_test, y_train, y_test = train_test_split(files, np.array(targets).astype(np.float16), 
                                                            test_size=0.25, random_state=10, shuffle=True)

        if split == "train":
            self.data = [Image.open(f) for f in X_train]
            self.targets = y_train
        else:
            self.data = [Image.open(f) for f in X_test]
            self.targets = y_test
        
        #import IPython;IPython.embed();raise Exception

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if len(np.array(image).shape)!=3:
            #print(np.array(image).shape)
            image = np.stack((np.array(image),)*3, axis=-1)
            image = Image.fromarray(image)

        if self.transform is not None:
            if self.albumentation:
                image = self.transform(image=image)["image"]
            else:
                image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)


class PosterRegressionDataModule(BaseDataModule):
    def __init__(self, **params):
        super(PosterRegressionDataModule, self).__init__(**params)
    
    def setup(self, stage: str):

        albumentations = "albumentations" in str(self.train_transforms.__class__)
        self.train_dataset = PosterRegressionData(self.root, split="train", transform=self.train_transforms, 
                                                  albumentation=albumentations)
        self.val_dataset = PosterRegressionData(self.root, split="val", transform=self.test_transforms, 
                                                albumentation=albumentations)
