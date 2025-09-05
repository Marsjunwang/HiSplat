from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor

T_cfg = TypeVar("T_cfg")
T_enhancer = TypeVar("T_enhancer")


class EnhancerVisualizer(ABC, Generic[T_cfg, T_enhancer]):
    cfg: T_cfg
    enhancer: T_enhancer

    def __init__(self, cfg: T_cfg, enhancer: T_enhancer) -> None:
        self.cfg = cfg
        self.enhancer = enhancer
        
    @abstractmethod
    def visualize(
        self,
        context: dict,
        global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"]]:
        pass