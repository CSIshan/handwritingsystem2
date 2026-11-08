try:
    from .model   import HandwritingModel
    from .trainer import Trainer, run_training
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config  import TrainConfig
from .prep    import prepare_dataset

__all__ = [
    "TrainConfig", "prepare_dataset",
]
if HAS_TORCH:
    __all__ += ["HandwritingModel", "Trainer", "run_training"]
