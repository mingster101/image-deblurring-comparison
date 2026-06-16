from __future__ import annotations
from abc import ABC, abstractmethod
from PIL import Image


class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.loaded = False
        self.load_error: str | None = None
        self.device = 'cpu'

    def _init_device(self):
        import config
        if config.DEVICE == 'auto':
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = 'cpu'
        else:
            self.device = config.DEVICE

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Image.Image:
        pass
