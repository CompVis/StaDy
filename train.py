"""Train StaDy on procedurally generated moving shapes."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from model import StaDy, LOSS_WEIGHTS
from moving_shapes import SyntheticShapeVideoDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/moving-shapes"))
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1, help="Per GPU")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--motion-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--model-config", type=Path, help="JSON architecture overrides")
    args = parser.parse_args()
    assert args.steps > 0 and args.batch_size > 0
    assert args.image_size >= 32 and args.image_size % 16 == 0
    assert args.save_every > 0 and args.log_every > 0 and args.warmup >= 0
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    payload = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if payload and "training" in payload:
        for key, value in payload["training"].items():
            setattr(args, key, value)
    config = dict(motion_dim=args.motion_dim)
    if args.model_config:
        config.update(json.loads(args.model_config.read_text()))
    if payload:
        config = payload["config"]
    model = StaDy(**config).to(device).train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    start = 0
    if payload:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start = payload["step"]
        assert args.steps > start, "--steps must exceed the checkpoint step"
    # Synthetic examples are indexed independently across ranks and continuation steps.
    dataset = SyntheticShapeVideoDataset(
        length=args.steps * args.batch_size * world,
        num_frames=2,
        image_size=args.image_size,
        min_size=args.image_size // 16,
        max_size=args.image_size // 4,
        max_speed=args.image_size / 2,
        max_angular_speed=120,
        motion_types=("bounce",),
        appearance_modes=("angular",),
        compact_motion_info=True,
        deterministic=True,
        seed=args.seed,
    )
    # Start the sampler at the next global batch without rendering skipped examples.
    indices = range(rank + start * args.batch_size * world, len(dataset), world)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=indices,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    wrapped = (
        DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        if world > 1
        else model
    )
    torch.manual_seed(args.seed + rank + start)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "config.json").write_text(json.dumps(model.config, indent=2) + "\n")
    for step, batch in enumerate(loader, start=start + 1):
        optimizer.zero_grad(set_to_none=True)
        lr = args.lr * min(step / max(args.warmup, 1), 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            losses, _ = wrapped(batch["x"].to(device, non_blocking=True))
            loss = sum(LOSS_WEIGHTS[k] * v.mean() for k, v in losses.items())
        assert torch.isfinite(loss).item(), f"Non-finite loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if rank == 0 and (step == start + 1 or step % args.log_every == 0):
            print(
                json.dumps(
                    dict(step=step, loss=loss.item(), **{k: v.mean().item() for k, v in losses.items()})
                ),
                flush=True,
            )
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            checkpoint = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                config=model.config,
                step=step,
                training={
                    k: getattr(args, k)
                    for k in ("lr", "warmup", "precision", "seed", "image_size", "batch_size")
                },
            )
            temporary = args.output / "latest.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "latest.pt")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
