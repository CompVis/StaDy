"""Render the Figure 1 transfer using the pretrained CelebV Torch Hub model."""

import argparse
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def read_image(path):
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).float().div(127.5).sub(1)


def as_image(tensor):
    return Image.fromarray(tensor.detach().cpu().clamp(-1, 1).add(1).mul(127.5).round().byte().numpy())


def render_panel(examples, font_path=None):
    """Lay out saved frames with centered Times New Roman Bold labels."""
    if font_path is None:
        font_path = subprocess.check_output(
            ["fc-match", "Times New Roman:style=Bold", "-f", "%{file}"], text=True
        ).strip()
    font = ImageFont.truetype(str(font_path), size=48)
    assert font.getname() == (
        "Times New Roman",
        "Bold",
    ), "Install Times New Roman Bold or pass --font /path/to/Timesbd.TTF"
    labels = ["Driving start", "Driving end", "Recipient start", "StaDy transfer"]
    files = ["driving_start.png", "driving_end.png", "start_frame.png", "prediction.png"]
    width, header_height = 4 * 256 + 5 * 12, 48
    # Supersample only the typography; preserve every image pixel.
    header = Image.new("RGB", (width * 2, header_height * 2), "white")
    draw = ImageDraw.Draw(header)
    for i, label in enumerate(labels):
        center = 12 + i * 268 + 128
        draw.text((center * 2, 25 * 2), label, fill="black", font=font, anchor="mm")
    canvas = Image.new("RGB", (width, header_height + 256 + 12), "white")
    canvas.paste(header.resize((width, header_height), Image.Resampling.LANCZOS), (0, 0))
    for i, filename in enumerate(files):
        canvas.paste(Image.open(examples / filename).convert("RGB"), (12 + i * 268, header_height))
    canvas.save(examples / "transfer.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font", type=Path, help="Path to Times New Roman Bold")
    parser.add_argument("--render-only", action="store_true", help="Restyle saved frames without inference")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    examples = root / "examples"
    if args.render_only:
        render_panel(examples, args.font)
        return
    model = torch.hub.load(str(root), "celebv", source="local", device=args.device)
    video = torch.stack(
        [read_image(examples / "driving_start.png"), read_image(examples / "driving_end.png")]
    )[None].to(args.device)
    source = read_image(examples / "start_frame.png")[None].to(args.device)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        motion = model.encode_motion(video)
        prediction = model.transfer(source, motion[:, 0], steps=args.steps)
    assert prediction.shape == source.shape and prediction.isfinite().all()
    as_image(prediction[0]).save(examples / "prediction.png")
    render_panel(examples, args.font)
    print(f'Saved {examples / "transfer.png"}', flush=True)


if __name__ == "__main__":
    main()
