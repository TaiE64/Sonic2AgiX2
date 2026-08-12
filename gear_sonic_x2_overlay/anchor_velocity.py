"""Pure Torch helpers for command-space anchor velocity tracking."""

from __future__ import annotations

import torch


def heading_frame_velocity(quaternion_wxyz: torch.Tensor, velocity_w: torch.Tensor) -> torch.Tensor:
    """Rotate a world-frame vector into the quaternion's yaw-only frame."""
    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    vx, vy, vz = velocity_w.unbind(dim=-1)
    return torch.stack(
        (cosine * vx + sine * vy, -sine * vx + cosine * vy, vz),
        dim=-1,
    )


def planar_velocity_tracking_reward(
    reference_quat_wxyz: torch.Tensor,
    reference_velocity_w: torch.Tensor,
    robot_quat_wxyz: torch.Tensor,
    robot_velocity_w: torch.Tensor,
    std: float,
) -> torch.Tensor:
    reference_velocity_b = heading_frame_velocity(
        reference_quat_wxyz, reference_velocity_w
    )
    robot_velocity_b = heading_frame_velocity(robot_quat_wxyz, robot_velocity_w)
    error = reference_velocity_b[..., :2] - robot_velocity_b[..., :2]
    return torch.exp(-error.square().sum(dim=-1) / (std * std))


def yaw_rate_tracking_reward(
    reference_quat_wxyz: torch.Tensor,
    reference_angular_velocity_w: torch.Tensor,
    robot_quat_wxyz: torch.Tensor,
    robot_angular_velocity_w: torch.Tensor,
    std: float,
) -> torch.Tensor:
    reference_rate = heading_frame_velocity(
        reference_quat_wxyz, reference_angular_velocity_w
    )[..., 2]
    robot_rate = heading_frame_velocity(robot_quat_wxyz, robot_angular_velocity_w)[..., 2]
    error = reference_rate - robot_rate
    return torch.exp(-error.square() / (std * std))
