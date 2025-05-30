from typing import TypeVar
import torch

T = TypeVar('T', bound=torch.Tensor)

# simple alias used only for type annotations
Float = T 