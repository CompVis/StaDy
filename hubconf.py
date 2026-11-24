"""The two selected pretrained StaDy models."""

from pathlib import Path as _Path
import torch as _torch
from huggingface_hub import hf_hub_download as _hf_hub_download

dependencies = ["torch", "torchvision", "einops", "diffusers", "huggingface_hub"]


def _load(name, model, device):
    path = _Path(__file__).resolve().parent / "checkpoints" / f"{name}.pt"
    if not path.is_file():
        path = _hf_hub_download(
            repo_id="CompVis/StaDy",
            filename=f"{name}.pt",
            revision="checkpoints-v1",
        )
    state = _torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def openworld(device="cpu"):
    """openworld: 1536-D motion, regularizer weight 2, step 430000."""
    from model import StaDy

    return _load("openworld", StaDy(motion_dim=768, num_motion_queries=2), device)


def celebv(device="cpu"):
    """CelebV-HQ: 64 x 8 motion tokens, regularizer weight 0.5, step 400000."""
    from model import CelebVStaDy

    return _load("celebv", CelebVStaDy(), device)
