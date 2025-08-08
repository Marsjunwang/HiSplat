from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Sequence

from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.types import BatchedViews, DataShim

T = TypeVar("T")



class Enhancer(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg
    @abstractmethod
    def pose_enhance(self, context: BatchedViews) -> BatchedViews:
        pass
    
    @abstractmethod
    def feat_enhance(self, context: BatchedViews) -> BatchedViews:
        pass
    
    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        features: Sequence,
    ) -> tuple[BatchedViews, Sequence]:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x
    
class PoseEnhancer(nn.Module, ABC, Generic[T]):
    cfg: T
    
    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg
    
    @abstractmethod
    def forward(self, context: BatchedViews, features: Sequence[Float[Tensor, "b c h w"]]) -> BatchedViews:
        pass
    
class FeatEnhancer(nn.Module, ABC, Generic[T]):
    cfg: T
    
    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg
    
    @abstractmethod
    def forward(self, context: BatchedViews, features: Sequence[Float[Tensor, "b c h w"]]) -> BatchedViews:
        pass
