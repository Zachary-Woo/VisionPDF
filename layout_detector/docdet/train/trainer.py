"""
DocDet Trainer: single-GPU training loop with AMP, cosine LR,
AdamW, gradient clipping, checkpointing, and TensorBoard logging.

Design notes
------------
* Device-agnostic: uses CUDA when available, falls back to CPU.
  The same script runs locally and on Colab unchanged.
* AMP (``torch.cuda.amp``) is enabled on CUDA devices and silently
  disabled on CPU.
* Checkpoints are written every N optimisation steps AND at the
  end of each epoch.  ``--resume-from`` restores optimizer +
  scheduler + AMP scaler state so training is bit-for-bit continuable.
* The trainer accepts any ``torch.utils.data.DataLoader`` whose
  ``collate_fn`` emits the schema from ``docdet_collate``.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..model.loss import DocDetLoss, compute_fcos_targets
from ..model.model import DocDet
from .config import TrainerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup -> cosine decay
# ---------------------------------------------------------------------------

def _warmup_cosine_schedule(
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> Callable[[int], float]:
    """Return a lambda(step) -> LR multiplier."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return fn


# ---------------------------------------------------------------------------
# Feature-map sizes derived from target image size.
# The backbone is stride 4/8/16 so this is a pure arithmetic helper.
# ---------------------------------------------------------------------------

def _feat_sizes_for(
    target_size: Tuple[int, int]
) -> Dict[str, Tuple[int, int]]:
    H, W = target_size
    return {
        "p2": (H // 4, W // 4),
        "p3": (H // 8, W // 8),
        "p4": (H // 16, W // 16),
    }


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """
    Single-GPU trainer for DocDet.

    Parameters
    ----------
    model      : DocDet instance.
    criterion  : DocDetLoss instance.
    train_loader : DataLoader yielding docdet_collate batches.
    config     : TrainerConfig dataclass.
    val_loader : optional DataLoader for held-out validation.
    eval_fn    : optional callable ``eval_fn(model) -> dict`` returning
                 metrics to log (e.g. mAP@0.5, table AP).  Executed
                 every ``eval_every_n_epochs`` epochs.
    """

    def __init__(
        self,
        model: DocDet,
        criterion: DocDetLoss,
        train_loader: DataLoader,
        config: TrainerConfig,
        val_loader: Optional[DataLoader] = None,
        eval_fn: Optional[Callable[[DocDet], Dict[str, float]]] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.eval_fn = eval_fn

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        if (
            os.environ.get("DOCDET_COMPILE", "0") == "1"
            and hasattr(torch, "compile")
            and self.device.type == "cuda"
        ):
            logger.info("torch.compile enabled (Ampere+ GPU detected)")
            self.model = torch.compile(self.model)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.steps_per_epoch = max(len(train_loader), 1)
        self.total_steps = self.steps_per_epoch * config.epochs
        min_lr_ratio = config.min_learning_rate / max(config.learning_rate, 1e-12)
        self.scheduler = LambdaLR(
            self.optimizer,
            _warmup_cosine_schedule(
                warmup_steps=config.warmup_steps,
                total_steps=self.total_steps,
                min_lr_ratio=min_lr_ratio,
            ),
        )

        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.global_step = 0
        self.epoch = 0

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._writer = self._build_tensorboard_writer()

        if config.resume_from is not None:
            self.load_checkpoint(config.resume_from)

    # --- utils ----------------------------------------------------------

    def _build_tensorboard_writer(self):
        """Return a SummaryWriter or None if tensorboard isn't installed."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning("tensorboard not installed; logs will be stdout only")
            return None
        return SummaryWriter(log_dir=str(self.save_dir / "tb" / self.config.run_name))

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    # --- checkpointing --------------------------------------------------

    def save_checkpoint(self, name: str) -> Path:
        """Serialise model + optimizer + scheduler state to disk."""
        path = self.save_dir / f"{name}.pt"
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else None,
        }
        torch.save(payload, path)
        logger.info("checkpoint saved to %s", path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Restore training state from a checkpoint file."""
        logger.info("resuming from %s", path)
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        if "scaler" in payload:
            self.scaler.load_state_dict(payload["scaler"])
        self.global_step = payload.get("global_step", 0)
        self.epoch = payload.get("epoch", 0)

    # --- one training step ---------------------------------------------

    def _prepare_targets(
        self,
        boxes_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        feat_sizes: Dict[str, Tuple[int, int]],
    ) -> List[Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Build per-image FCOS targets on the trainer device."""
        targets_per_image = []
        for boxes, labels in zip(boxes_list, labels_list):
            if boxes.numel() == 0:
                boxes = boxes.reshape(0, 4)
            boxes = boxes.to(self.device)
            labels = labels.to(self.device)
            targets_per_image.append(
                compute_fcos_targets(boxes, labels, feat_sizes)
            )
        return targets_per_image

    def _training_step(self, batch) -> Dict[str, float]:
        """Forward + backward + optimizer step for one batch."""
        images = batch["images"].to(self.device, non_blocking=True)
        feat_sizes = _feat_sizes_for(self.config.target_size)

        targets_per_image = self._prepare_targets(
            batch["boxes"], batch["labels"], feat_sizes,
        )

        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(images)
            loss, stats = self.criterion(outputs, targets_per_image)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_max_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_max_norm
            )
            self.optimizer.step()

        self.scheduler.step()
        self.global_step += 1

        stats["loss"] = float(loss.detach().item())
        stats["lr"] = self.optimizer.param_groups[0]["lr"]
        return stats

    # --- main entry point ----------------------------------------------

    def train(self) -> None:
        """Run the full training loop per ``config.epochs``."""
        if self.config.freeze_backbone_stages > 0:
            self.model.freeze_backbone_stages(self.config.freeze_backbone_stages)

        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            self.model.train()

            epoch_start = time.time()
            running_loss = 0.0
            running_cls = 0.0
            running_reg = 0.0
            running_cent = 0.0

            for step, batch in enumerate(self.train_loader):
                stats = self._training_step(batch)
                running_loss += stats["loss"]
                running_cls += stats["cls_loss"]
                running_reg += stats["reg_loss"]
                running_cent += stats["cent_loss"]

                if (step + 1) % self.config.log_every_n_steps == 0:
                    avg_loss = running_loss / self.config.log_every_n_steps
                    avg_cls = running_cls / self.config.log_every_n_steps
                    avg_reg = running_reg / self.config.log_every_n_steps
                    avg_cent = running_cent / self.config.log_every_n_steps
                    logger.info(
                        "epoch %d step %d/%d | loss %.4f (cls %.4f reg %.4f cent %.4f) lr %.2e",
                        epoch, step + 1, self.steps_per_epoch,
                        avg_loss, avg_cls, avg_reg, avg_cent,
                        stats["lr"],
                    )
                    self._log_scalar("train/loss", avg_loss, self.global_step)
                    self._log_scalar("train/cls_loss", avg_cls, self.global_step)
                    self._log_scalar("train/reg_loss", avg_reg, self.global_step)
                    self._log_scalar("train/cent_loss", avg_cent, self.global_step)
                    self._log_scalar("train/lr", stats["lr"], self.global_step)
                    running_loss = running_cls = running_reg = running_cent = 0.0

                if (
                    self.config.checkpoint_every_n_steps > 0
                    and self.global_step % self.config.checkpoint_every_n_steps == 0
                ):
                    self.save_checkpoint(f"{self.config.run_name}_step{self.global_step}")

            logger.info(
                "epoch %d complete in %.1f s",
                epoch, time.time() - epoch_start,
            )
            self.save_checkpoint(f"{self.config.run_name}_epoch{epoch}")

            if (
                self.eval_fn is not None
                and (epoch + 1) % self.config.eval_every_n_epochs == 0
            ):
                logger.info("running eval callback ...")
                self.model.eval()
                metrics = self.eval_fn(self.model)
                self.model.train()
                for name, value in metrics.items():
                    self._log_scalar(f"val/{name}", value, self.global_step)
                    logger.info("val/%s = %.4f", name, value)

        self.save_checkpoint(f"{self.config.run_name}_final")
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
