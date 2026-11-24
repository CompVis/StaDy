import math
import random
from functools import partial
from typing import Callable, Dict, Tuple, Optional

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import default_collate



class SyntheticShapeVideoDataset(Dataset):
    def __init__(
        self,
        length: int,
        num_frames: int = 16,
        image_size: int = 64,
        channels: int = 3,
        shape_types: Tuple[str, ...] = ("circle", "square", "triangle", "ellipse"),
        motion_types: Tuple[str, ...] = ("linear", "bounce", "accelerated", "circular"),
        appearance_modes: Tuple[str, ...] = ("solid", "linear", "angular", "stripes"),
        min_size: int = 6,
        max_size: int = 14,
        max_speed: float = 3.0,
        max_angular_speed: float = 12.0,
        deterministic: bool = False,
        return_uint8: bool = False,
        compact_motion_info: bool = False,
        render_transfer_video: bool = False,
        framewise_appearance: float = 0.0,
        transform: Optional[Callable] = None,
        seed: int | None = None,
    ):
        self.length = length
        self.num_frames = num_frames
        self.image_size = image_size
        self.channels = channels
        self.shape_types = shape_types
        self.motion_types = motion_types
        self.appearance_modes = appearance_modes
        self.min_size = min_size
        self.max_size = max_size
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.deterministic = deterministic
        self.return_uint8 = return_uint8
        self.compact_motion_info = compact_motion_info
        self.render_transfer_video = render_transfer_video
        self.framewise_appearance = framewise_appearance
        self.transform = transform
        self.seed = seed

        if channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")
        assert 0.0 <= framewise_appearance <= 1.0
        if framewise_appearance > 0.0:
            assert len(appearance_modes) > 1, "framewise_appearance requires at least two appearance modes"

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = self._make_rng(idx)

        H = W = self.image_size
        # -------------------------
        # 1) Sample appearance params
        # -------------------------
        shape_type = rng.choice(self.shape_types)
        size = rng.randint(self.min_size, self.max_size)
        appearance_mode = rng.choice(self.appearance_modes)
        appearance_modes_per_frame = [appearance_mode] * self.num_frames
        if rng.random() < self.framewise_appearance:
            next_appearance_mode = self.appearance_modes[
                (self.appearance_modes.index(appearance_mode) + 1) % len(self.appearance_modes)
            ]
            appearance_modes_per_frame[1:] = [next_appearance_mode] * (self.num_frames - 1)

        if self.channels == 1:
            color1 = (rng.randint(100, 255),)
            color2 = (rng.randint(100, 255),)
        else:
            color1 = (
                rng.randint(64, 255),
                rng.randint(64, 255),
                rng.randint(64, 255),
            )
            color2 = (
                rng.randint(64, 255),
                rng.randint(64, 255),
                rng.randint(64, 255),
            )

        # background color
        bg_color = self._sample_distinct_background_color(
            rng,
            object_colors=[color1, color2],
            min_color_dist=120.0,
            pastel_background=True,
        )

        # Rotation is meaningful for non-circular shapes
        angle0 = rng.uniform(0.0, 360.0)
        angular_velocity = rng.uniform(-self.max_angular_speed, self.max_angular_speed)

        # Ellipse-specific aspect ratio
        min_aspect = 0.6
        max_aspect = 1.6
        aspect = rng.uniform(min_aspect, max_aspect)

        # -------------------------
        # 2) Sample motion params
        # -------------------------
        motion_type = rng.choice(self.motion_types)

        # Safe margin so object starts inside frame
        radius_bound = int(math.ceil(size * max(1.0, aspect))) + 2

        x0 = rng.uniform(radius_bound, W - radius_bound)
        y0 = rng.uniform(radius_bound, H - radius_bound)

        vx0 = rng.uniform(-self.max_speed, self.max_speed)
        vy0 = rng.uniform(-self.max_speed, self.max_speed)

        # Avoid nearly static samples unless desired
        if abs(vx0) + abs(vy0) < 0.5:
            vx0 = 1.0 if vx0 >= 0 else -1.0
            vy0 = 1.0 if vy0 >= 0 else -1.0

        max_acceleration = 0.15
        ax = rng.uniform(-max_acceleration, max_acceleration)
        ay = rng.uniform(-max_acceleration, max_acceleration)

        # Circular motion parameters
        orbit_radius = rng.uniform(6.0, min(H, W) * 0.25)
        omega = rng.uniform(-0.25, 0.25)  # rad per frame
        phase0 = rng.uniform(0.0, 2.0 * math.pi)

        # Keep center for circular motion in safe area
        cx = rng.uniform(radius_bound + orbit_radius, W - radius_bound - orbit_radius)
        cy = rng.uniform(radius_bound + orbit_radius, H - radius_bound - orbit_radius)

        appearance_render_info = {
            "shape_type": shape_type,
            "size": size,
            "appearance_mode": appearance_mode,
            "appearance_modes_per_frame": appearance_modes_per_frame,
            "color1": color1,
            "color2": color2,
            "bg_color": bg_color,
            "aspect": aspect,
        }
        pose_render_info = self._generate_pose(
            motion_type=motion_type,
            radius_bound=radius_bound,
            x0=x0,
            y0=y0,
            vx0=vx0,
            vy0=vy0,
            ax=ax,
            ay=ay,
            orbit_radius=orbit_radius,
            omega=omega,
            phase0=phase0,
            cx=cx,
            cy=cy,
            angle0=angle0,
            angular_velocity=angular_velocity,
        )
        centers, angles = pose_render_info
        video = self._render_video(appearance_render_info, pose_render_info)

        video_tensor = self._video_to_output_tensor(video)
        video_tensor_aug = (
            self._video_to_output_tensor(video, apply_transform=True) if self.transform is not None else None
        )

        # static appearance info
        appearance = {
            "shape_type": self._one_hot(self.shape_types.index(shape_type), len(self.shape_types)),
            "appearance_mode": (
                torch.stack(
                    [
                        self._one_hot(self.appearance_modes.index(mode), len(self.appearance_modes))
                        for mode in appearance_modes_per_frame
                    ],
                    dim=0,
                )
                if self.framewise_appearance > 0.0
                else self._one_hot(self.appearance_modes.index(appearance_mode), len(self.appearance_modes))
            ),
            "size": torch.tensor([float((size - self.min_size) / (self.max_size - self.min_size))]),
            "aspect": torch.tensor([float((aspect - min_aspect) / (max_aspect - min_aspect))]),
            "color1": torch.tensor(color1, dtype=torch.float32).div(255.0),
            "color2": torch.tensor(color2, dtype=torch.float32).div(255.0),
            "background_color": torch.tensor(bg_color, dtype=torch.float32).div(255.0),
        }

        # frame-wise pose info
        centers = torch.from_numpy(centers)
        centers_x = centers[:, 0].div(W - 1)
        centers_y = centers[:, 1].div(H - 1)
        angles_deg = torch.from_numpy(angles)
        angles_rad = angles_deg.mul(math.pi).div(180.0)
        framewise_pose = {
            "positions_x": centers_x,
            "positions_y": centers_y,
            "angles_sin": angles_rad.sin(),
            "angles_cos": angles_rad.cos(),
        }

        # frame-wise motion info
        velocities_x = centers_x[1:] - centers_x[:-1]
        velocities_y = centers_y[1:] - centers_y[:-1]
        angular_velocities_rad = angles_rad[1:] - angles_rad[:-1]
        if self.compact_motion_info:
            # Compact normalized motion targets are easier for the minimal extractor
            # than raw frame-wise deltas and angle sin/cos.
            vx_scale = max(self.max_speed / max(W - 1, 1), 1e-6)
            vy_scale = max(self.max_speed / max(H - 1, 1), 1e-6)
            omega_scale = max(math.radians(self.max_angular_speed), 1e-6)

            vx_norm = (velocities_x.mean() / vx_scale).clamp(-1.0, 1.0)
            vy_norm = (velocities_y.mean() / vy_scale).clamp(-1.0, 1.0)
            omega_norm = (angular_velocities_rad.mean() / omega_scale).clamp(-1.0, 1.0)
            speed_norm = (vx_norm.square() + vy_norm.square()).sqrt().div(math.sqrt(2.0)).clamp(0.0, 1.0)

            framewise_motion = {
                "velocities_x": vx_norm.unsqueeze(0),
                "velocities_y": vy_norm.unsqueeze(0),
                "angular_velocities_sin": omega_norm.unsqueeze(0),
                "angular_velocities_cos": speed_norm.unsqueeze(0),
            }
        else:
            framewise_motion = {
                "velocities_x": velocities_x,
                "velocities_y": velocities_y,
                "angular_velocities_sin": angular_velocities_rad.sin(),
                "angular_velocities_cos": angular_velocities_rad.cos(),
            }

        sample = {
            "x": video_tensor,  # [T, H, W, C] in (-1, 1)
            "appearance_info": appearance,
            "pose_info": framewise_pose,
            "motion_info": framewise_motion,
        }
        if video_tensor_aug is not None:
            sample["x_aug"] = video_tensor_aug
        if self.render_transfer_video:
            sample["_appearance_render_info"] = appearance_render_info
            sample["_pose_render_info"] = pose_render_info
        return sample

    def collate_fn(self, samples):
        batch = default_collate(
            [{k: v for k, v in sample.items() if not k.startswith("_")} for sample in samples]
        )
        if self.render_transfer_video:
            transfer_videos = []
            for sample, appearance_sample in zip(samples, samples[-1:] + samples[:-1]):
                transfer_video = self._render_video(
                    appearance_sample["_appearance_render_info"],
                    self._transfer_pose_delta(
                        source_pose_render_info=sample["_pose_render_info"],
                        target_pose_render_info=appearance_sample["_pose_render_info"],
                        target_appearance_render_info=appearance_sample["_appearance_render_info"],
                    ),
                )
                transfer_videos.append(self._video_to_output_tensor(transfer_video))
            batch["transfer_video"] = torch.stack(transfer_videos, dim=0)
        return batch

    def _transfer_pose_delta(
        self,
        source_pose_render_info: tuple[np.ndarray, np.ndarray],
        target_pose_render_info: tuple[np.ndarray, np.ndarray],
        target_appearance_render_info: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        source_centers, source_angles = source_pose_render_info
        target_centers, target_angles = target_pose_render_info
        centers = target_centers[:1] + (source_centers - source_centers[:1])
        margin = (
            int(
                math.ceil(
                    target_appearance_render_info["size"] * max(1.0, target_appearance_render_info["aspect"])
                )
            )
            + 2
        )
        centers = centers.copy()
        for i, (x, y) in enumerate(centers):
            centers[i] = self._clip_inside(float(x), float(y), margin, self.image_size, self.image_size)
        angles = target_angles[:1] + (source_angles - source_angles[:1])
        return centers, angles

    def _video_to_output_tensor(self, video: np.ndarray, apply_transform: bool = False) -> torch.Tensor:
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()
        if apply_transform:
            assert self.transform is not None
            video_tensor, _ = self.transform(video_tensor)

        if self.return_uint8:
            return video_tensor.permute(0, 2, 3, 1)

        video_tensor = video_tensor.float() / 255.0
        return video_tensor.permute(0, 2, 3, 1).mul(2).sub(1)

    def _generate_pose(
        self,
        motion_type: str,
        radius_bound: int,
        x0: float,
        y0: float,
        vx0: float,
        vy0: float,
        ax: float,
        ay: float,
        orbit_radius: float,
        omega: float,
        phase0: float,
        cx: float,
        cy: float,
        angle0: float,
        angular_velocity: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        H = W = self.image_size
        centers = np.empty((self.num_frames, 2), dtype=np.float32)
        angles = np.empty(self.num_frames, dtype=np.float32)

        x, y = x0, y0
        vx, vy = vx0, vy0

        for t in range(self.num_frames):
            if motion_type == "linear":
                xt = x0 + vx0 * t
                yt = y0 + vy0 * t
                xt, yt = self._clip_inside(xt, yt, radius_bound, W, H)

            elif motion_type == "bounce":
                if t > 0:
                    x += vx
                    y += vy

                    if x < radius_bound:
                        x = radius_bound + (radius_bound - x)
                        vx = -vx
                    elif x > W - radius_bound:
                        x = (W - radius_bound) - (x - (W - radius_bound))
                        vx = -vx

                    if y < radius_bound:
                        y = radius_bound + (radius_bound - y)
                        vy = -vy
                    elif y > H - radius_bound:
                        y = (H - radius_bound) - (y - (H - radius_bound))
                        vy = -vy

                xt, yt = x, y

            elif motion_type == "accelerated":
                xt = x0 + vx0 * t + 0.5 * ax * (t**2)
                yt = y0 + vy0 * t + 0.5 * ay * (t**2)
                xt, yt = self._clip_inside(xt, yt, radius_bound, W, H)

            elif motion_type == "circular":
                theta = phase0 + omega * t
                xt = cx + orbit_radius * math.cos(theta)
                yt = cy + orbit_radius * math.sin(theta)

            else:
                raise ValueError(f"Unknown motion_type: {motion_type}")

            centers[t] = (xt, yt)
            angles[t] = angle0 + angular_velocity * t

        return centers, angles

    def _render_video(
        self,
        appearance_render_info: dict,
        pose_render_info: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        H = W = self.image_size
        centers, angles = pose_render_info
        appearance_modes_per_frame = appearance_render_info.get(
            "appearance_modes_per_frame",
            [appearance_render_info["appearance_mode"]] * len(centers),
        )
        frames = []

        for (xt, yt), angle_t, appearance_mode in zip(centers, angles, appearance_modes_per_frame):
            frame = self._make_blank_frame(H, W, appearance_render_info["bg_color"])
            mask = self._draw_shape_mask(
                H=H,
                W=W,
                shape_type=appearance_render_info["shape_type"],
                center=(float(xt), float(yt)),
                size=appearance_render_info["size"],
                angle_deg=float(angle_t),
                aspect=appearance_render_info["aspect"],
            )

            self._fill_mask_with_appearance(
                frame=frame,
                mask=mask,
                center=(float(xt), float(yt)),
                angle_deg=float(angle_t),
                appearance_mode=appearance_mode,
                color1=appearance_render_info["color1"],
                color2=appearance_render_info["color2"],
            )

            frames.append(frame)

        video = np.stack(frames, axis=0)  # [T, H, W, C] or [T, H, W]
        if self.channels == 1:
            video = video[..., None]  # [T, H, W, 1]
        return video

    def _make_rng(self, idx: int) -> random.Random:
        if self.seed is not None:
            return random.Random(self.seed + idx)
        if self.deterministic:
            return random.Random(idx)
        return random.Random()

    def _make_blank_frame(self, H: int, W: int, background_color) -> np.ndarray:
        if self.channels == 1:
            return np.full((H, W), background_color[0], dtype=np.uint8)
        return np.full((H, W, 3), background_color, dtype=np.uint8)

    @staticmethod
    def _clip_inside(
        x: float,
        y: float,
        margin: int,
        W: int,
        H: int,
    ) -> Tuple[float, float]:
        x = min(max(x, margin), W - margin)
        y = min(max(y, margin), H - margin)
        return x, y

    def _sample_distinct_background_color(
        self,
        rng,
        object_colors,
        min_color_dist: float = 100.0,
        max_tries: int = 100,
        pastel_background: bool = False,
    ):
        """
        Sample a background color that stays sufficiently distinct from all object colors.

        Args:
            object_colors: iterable of RGB tuples/lists in [0, 255]
            min_color_dist: minimum Euclidean RGB distance
            pastel_background: if True, prefer softer backgrounds

        Returns:
            bg_color: tuple(int, int, int)
        """
        object_colors = [np.array(c, dtype=np.float32) for c in object_colors]

        for _ in range(max_tries):
            if pastel_background:
                # softer / less aggressive colors
                bg = np.array(
                    [
                        rng.randint(80, 220),
                        rng.randint(80, 220),
                        rng.randint(80, 220),
                    ],
                    dtype=np.float32,
                )
            else:
                bg = np.array(
                    [
                        rng.randint(0, 255),
                        rng.randint(0, 255),
                        rng.randint(0, 255),
                    ],
                    dtype=np.float32,
                )

            dists = [np.linalg.norm(bg - c) for c in object_colors]
            if min(dists) >= min_color_dist:
                return tuple(bg.astype(np.uint8).tolist())

        # fallback: push away from average object color
        mean_obj = np.mean(object_colors, axis=0)
        bg = 255.0 - mean_obj
        bg = np.clip(bg, 0, 255)
        return tuple(bg.astype(np.uint8).tolist())

    def _draw_shape_mask(
        self,
        H: int,
        W: int,
        shape_type: str,
        center: Tuple[float, float],
        size: int,
        angle_deg: float,
        aspect: float,
    ) -> np.ndarray:
        mask = np.zeros((H, W), dtype=np.uint8)

        cx, cy = center
        cx_i, cy_i = int(round(cx)), int(round(cy))

        if shape_type == "circle":
            cv2.circle(mask, (cx_i, cy_i), size, 1, thickness=-1)

        elif shape_type == "square":
            pts = self._rotated_box_points(cx, cy, size * 2, size * 2, angle_deg)
            cv2.fillPoly(mask, [pts], 1)

        elif shape_type == "triangle":
            pts = self._rotated_triangle_points(cx, cy, size, angle_deg)
            cv2.fillPoly(mask, [pts], 1)

        elif shape_type == "ellipse":
            axis1 = int(round(size * aspect))
            axis2 = int(round(size / max(aspect, 1e-6)))
            axis1 = max(axis1, 1)
            axis2 = max(axis2, 1)
            cv2.ellipse(
                mask,
                center=(cx_i, cy_i),
                axes=(axis1, axis2),
                angle=angle_deg,
                startAngle=0,
                endAngle=360,
                color=1,
                thickness=-1,
            )

        else:
            raise ValueError(f"Unknown shape_type: {shape_type}")

        return mask

    def _fill_mask_with_appearance(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        center: Tuple[float, float],
        angle_deg: float,
        appearance_mode: str,
        color1: Tuple[int, int, int],
        color2: Optional[Tuple[int, int, int]] = None,
        stripe_width: float = 6.0,
    ):
        """
        Fill a shape mask with a parametric appearance pattern.

        appearance_mode:
            - "solid"
            - "linear"
            - "angular"
            - "stripes"
        """
        H, W = frame.shape[:2]
        cx, cy = center

        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        x = xs - cx
        y = ys - cy

        theta = np.deg2rad(angle_deg)
        c, s = np.cos(theta), np.sin(theta)

        # Object-aligned coordinates: texture rotates with the object
        u = x * c + y * s
        v = -x * s + y * c

        color1 = np.array(color1, dtype=np.float32)
        if color2 is None:
            color2 = (255, 255, 255)
        color2 = np.array(color2, dtype=np.float32)

        out = frame.astype(np.float32).copy()

        if appearance_mode == "solid":
            out[mask > 0] = color1

        elif appearance_mode == "linear":
            vals = u
            vals_obj = vals[mask > 0]
            vmin, vmax = vals_obj.min(), vals_obj.max()
            alpha = (vals - vmin) / (vmax - vmin + 1e-6)
            rgb = (1.0 - alpha[..., None]) * color1 + alpha[..., None] * color2
            out[mask > 0] = rgb[mask > 0]

        elif appearance_mode == "angular":
            ang = np.arctan2(v, u)  # object-aligned angular coordinate
            alpha = (ang + np.pi) / (2 * np.pi)
            rgb = (1.0 - alpha[..., None]) * color1 + alpha[..., None] * color2
            out[mask > 0] = rgb[mask > 0]

        elif appearance_mode == "stripes":
            phase = 2.0 * np.pi * u / stripe_width
            alpha = 0.5 * (np.sin(phase) + 1.0)
            rgb = (1.0 - alpha[..., None]) * color1 + alpha[..., None] * color2
            out[mask > 0] = rgb[mask > 0]

        else:
            raise ValueError(f"Unknown appearance_mode: {appearance_mode}")

        frame[...] = np.clip(out, 0, 255).astype(np.uint8)

    def _one_hot(self, idx: int, num_classes: int) -> torch.Tensor:
        v = torch.zeros(num_classes, dtype=torch.float32)
        v[idx] = 1.0
        return v

    @staticmethod
    def _rotated_box_points(
        cx: float,
        cy: float,
        w: float,
        h: float,
        angle_deg: float,
    ) -> np.ndarray:
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)

        corners = np.array(
            [
                [-w / 2, -h / 2],
                [w / 2, -h / 2],
                [w / 2, h / 2],
                [-w / 2, h / 2],
            ],
            dtype=np.float32,
        )

        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = corners @ R.T
        pts[:, 0] += cx
        pts[:, 1] += cy
        return np.round(pts).astype(np.int32)

    @staticmethod
    def _rotated_triangle_points(
        cx: float,
        cy: float,
        size: float,
        angle_deg: float,
    ) -> np.ndarray:
        # Equilateral-ish upright triangle centered around origin
        pts = np.array(
            [
                [0.0, -size],
                [-0.866 * size, 0.5 * size],
                [0.866 * size, 0.5 * size],
            ],
            dtype=np.float32,
        )

        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)

        pts = pts @ R.T
        pts[:, 0] += cx
        pts[:, 1] += cy
        return np.round(pts).astype(np.int32)


def SyntheticShapeVideoDataLoader(
    num_workers,
    batch_size,
    shuffle,
    prefetch_factor=None,
    seed: int | None = None,
    render_transfer_video_val: bool | None = None,
    **kwargs,
):
    dataloader_kwargs = {}
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    if seed is not None:
        dataloader_kwargs["generator"] = torch.Generator().manual_seed(seed)

    val_kwargs = dict(kwargs)
    if render_transfer_video_val is not None:
        val_kwargs["render_transfer_video"] = render_transfer_video_val

    def make_dataloader(deterministic: bool, batch_size: int, seed: int | None = None):
        dataset_kwargs = val_kwargs if deterministic else kwargs
        dataset = SyntheticShapeVideoDataset(deterministic=deterministic, seed=seed, **dataset_kwargs)
        loader_kwargs = dict(dataloader_kwargs)
        if seed is not None:
            loader_kwargs["generator"] = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            **loader_kwargs,
        )

    return {
        "train_dataloader": partial(make_dataloader, deterministic=False, batch_size=batch_size, seed=seed),
        "val_dataloader": partial(make_dataloader, deterministic=True, batch_size=8, seed=seed),
    }
