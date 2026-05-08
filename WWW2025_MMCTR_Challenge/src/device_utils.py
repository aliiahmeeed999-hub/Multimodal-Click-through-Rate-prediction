"""Resolve CLI `--gpu` id to an effective FuxiCTR `gpu` index (>=0 = CUDA, -1 = CPU)."""
from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


def resolve_training_gpu(requested_gpu: int) -> int:
    if requested_gpu < 0:
        logger.info("Using CPU (gpu=%s).", requested_gpu)
        return -1
    if not torch.cuda.is_available():
        logger.warning(
            "GPU %s requested but CUDA is not available; falling back to CPU. "
            "Install a CUDA-enabled PyTorch build and drivers to use GPU.",
            requested_gpu,
        )
        return -1
    n = torch.cuda.device_count()
    if requested_gpu >= n:
        logger.warning(
            "GPU %s requested but only %s CUDA device(s) available; using cuda:0.",
            requested_gpu,
            n,
        )
        effective = 0
    else:
        effective = requested_gpu
    logger.info(
        "Using CUDA device %s: %s",
        effective,
        torch.cuda.get_device_name(effective),
    )
    return effective
