# Copyright (c) Facebook, Inc. and its affiliates.
# Ruchao Fan SPAPL 2022
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class NoamWarmupLRScheduleConfig(FairseqDataclass):
    warmup_steps: int = field(
        default=0,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    noam_factor: List[float] = II("optimization.lr")
    init_lr_scale: float = field(
        default=0.01,
        metadata={"help": "initial learning rate scale during warmup phase"},
    )
    max_update: float = II("optimization.max_update")
    lr: List[float] = II("optimization.lr")
    
@register_lr_scheduler("noam_warmup", dataclass=NoamWarmupLRScheduleConfig)
class NoamWarmupLRSchedule(FairseqLRScheduler):
    """NoamWarmup learning rate schedulr"""

    def __init__(self, cfg: NoamWarmupLRScheduleConfig, optimizer):
        super().__init__(cfg, optimizer)

        # calculate LR at each point
        self.peak_lr = cfg.lr[0]
        self.init_lr = cfg.init_lr_scale * cfg.lr[0]

        self.warmup_steps = cfg.warmup_steps

        # initial learning rate
        self.lr = self.init_lr
        self.optimizer.set_lr(self.lr)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        num_updates = max(num_updates, 1) #prevent 0
        factor = self.warmup_steps ** 0.5 * min(num_updates ** (-0.5), num_updates * self.warmup_steps ** (-1.5))
        self.lr = self.peak_lr * factor
        self.optimizer.set_lr(self.lr)

        return self.lr
