# Figure 1 motion transfer

- `driving.mp4`: original CelebV-HQ clip `1RRRM17-4Ok_2_0`, 102 frames at 24 fps.
- `driving_start.png`, `driving_end.png`: the exact saved evaluation inputs used
  for the driving pair in Figure 1 (matching video frames 14 and 90).
- `start_frame.png`: the original recipient image embedded at 256 × 256 in the
  paper figure; no figure annotations are included.
- `inputs.json`: input provenance and frame matching information.

Run `python example_transfer.py` from the release root to generate the transfer
with the CelebV checkpoint, 50 Euler steps, and seed 42.

The video and images are research example media from CelebV-HQ and the paper;
the software MIT license does not grant rights to the underlying media.

Panel labels use Times New Roman Bold. Install the font locally or provide
`--font /path/to/Timesbd.TTF`. To restyle saved frames without inference, run
`python example_transfer.py --render-only`.
