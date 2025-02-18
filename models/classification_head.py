import torch
import torch.nn as nn
from timm.layers import ClassifierHead


class ClassificationHead(nn.Module):
    def __init__(
        self, embed_dim, num_classes, dropout=0.1, patch_aggregation_method="avg"
    ):
        """
        Args:
            embed_dim (int): size of the embedding.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate applied before the output layer.
            patch_aggregation_method (string): "cls_token" for taking the class token, "avg" or "sum"
                                                for aggregating the individual token vectors
        """
        super(ClassificationHead, self).__init__()

        self.fc = ClassifierHead(embed_dim, num_classes, "", dropout)

        self.patch_aggregation_method = patch_aggregation_method

    def forward(self, x):

        if self.patch_aggregation_method is not None:
            if self.patch_aggregation_method == "cls_token":
                x = x[:, 0]
            elif self.patch_aggregation_method == "avg":
                x = x[:, 1:].mean(dim=1)
            elif self.patch_aggregation_method == "sum":
                x = x[:, 1:].sum(dim=1)

        x = self.fc(x)

        return x
