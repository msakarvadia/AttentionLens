from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    # Lightning arguments
    lr: float = field(default=1e-3)
    epochs: int = field(default=3)
    max_checkpoint_num: int = field(default=1)
    batch_size: int = field(default=1)
    num_nodes: int = field(default=1)
    mixed_precision: bool = field(default=True)
    # TODO: add argument that sets/fixes precision to 32- or 64-bit.
    checkpoint_mode: str = field(default="step")
    num_steps_per_checkpoint: int = field(default=5)
    checkpoint_dir: Path | str = field(default="checkpoint")
    accumulate_grad_batches: int = field(default=10)
    reload_checkpoint: Optional[Path | str] = field(default=None)
    stopping_delta: float = field(default=1e-7)
    stopping_patience: int = field(default=2)

    # AttentionLens-specific arguments
    model_name: str = field(default="gpt2")  # Choices are `OFFICIAL_MODEL_NAMES`
    layer_number: int = field(default=0)

    def __post_init__(self):
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.reload_checkpoint, str):
            self.checkpoint_dir = Path(self.reload_checkpoint)
