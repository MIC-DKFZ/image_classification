from .classification import ClassificationHead
from .mil.clam import CLAM_MB, CLAM_SB
from .regression import RegressionHead
from .video.framewise_decoder_1d import FramewiseDecoder1D

__all__ = ["ClassificationHead", "CLAM_SB", "CLAM_MB", "FramewiseDecoder1D", "RegressionHead"]
