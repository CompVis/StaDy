# Selected weights

Torch Hub exposes two fixed pretrained models hosted on
[Hugging Face](https://huggingface.co/CompVis/StaDy). Missing checkpoints are
downloaded automatically and cached by `huggingface_hub`. Downloads are pinned
to revision `checkpoints-v1`.

You can also place `openworld.pt` or `celebv.pt` in this folder to use a local copy.
Local checkpoint files take precedence over downloads and are excluded from Git.

| File / Hub entry | Training | Checkpoint | Motion representation | Regularizer weight |
| --- | --- | --- | --- | --- |
| `openworld.pt` / `openworld` | openworld | 430,000 | 2 × 768 | 2 |
| `celebv.pt` / `celebv` | CelebV-HQ | 400,000 | 64 × 8 | 0.5 |

openworld uses the high-capacity λ=0.2 paper setting (full reconstruction weight 10).
CelebV uses the high-capacity λ=0.5 setting with the strongest joint motion and
identity results in the CelebV capacity comparison. The paper's CelebV plotting
label for this model is 2048; the saved architecture contains 64 eight-dimensional
tokens. These are online model weights, matching the default evaluation path.
These fixed selections do not imply that one checkpoint wins every downstream metric.

The checkpoints total 4.01 GB (openworld: 3.37 GB; CelebV: 637 MB).
Checksums are recorded in [manifest.json](manifest.json).

Both checkpoints are public and can be downloaded without a Hugging Face login
or access token.
