from .classification import ClassificationHead
from .mil.clam import CLAM_MB, CLAM_SB
from .regression import RegressionHead

__all__ = ["ClassificationHead", "CLAM_SB", "CLAM_MB", "RegressionHead"]
