from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn

from ...dataset.types import BatchedViews, DataShim

T = TypeVar("T")



class Enhancer(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        deterministic: bool,
    ) -> BatchedViews:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x
