import torch
import torch.nn as nn
from timm.layers import ClassifierHead
from models.mil.abmil import ABMIL
from models.mil.chowder import Chowder
from models.mil.hiptmil import HIPTMIL


class ClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_classes,
        dropout=0.1,
        patch_aggregation_method="avg",
        cls_token_available=True,
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
        self.cls_token_available = cls_token_available

        if self.patch_aggregation_method == "abmil":
            self.mil_alg = ABMIL(
                embed_dim,
                out_features=num_classes,
                metadata_cols=0,
                d_model_attention=128,
                mlp_dropout=[dropout],
                mlp_hidden=[64],
            )
            self.fc = None
        elif self.patch_aggregation_method == "chowder_mil":
            self.mil_alg = Chowder(
                embed_dim,
                out_features=num_classes,
                metadata_cols=0,
                n_top=64,
                n_bottom=64,
                mlp_dropout=[dropout],
                mlp_hidden=[64],
            )
            self.fc = None
        elif self.patch_aggregation_method == "hiptmil":
            # batch size needs to be 1 for this method
            self.mil_alg = HIPTMIL(
                embed_dim,
                out_features=num_classes,
                metadata_cols=0,
            )
            self.fc = None

    def forward(self, x):

        if self.patch_aggregation_method is not None:
            if self.patch_aggregation_method == "cls_token":
                assert self.cls_token_available
                x = x[:, 0]
            elif self.patch_aggregation_method == "avg":
                x = x[:, 1:].mean(dim=1) if self.cls_token_available else x.mean(dim=1)
            elif self.patch_aggregation_method == "sum":
                x = x[:, 1:].sum(dim=1) if self.cls_token_available else x.sum(dim=1)

            elif "mil" in self.patch_aggregation_method:
                x = (
                    self.mil_alg(x[:, 1:])
                    if self.cls_token_available
                    else self.mil_alg(x)
                )

        if (
            self.patch_aggregation_method is not None
            and "mil" in self.patch_aggregation_method
        ):
            # MIL algorithms use their own classification heads
            return x
        else:
            x = self.fc(x)

        return x
