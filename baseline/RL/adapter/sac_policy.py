"""Plain-torch port of the MS-HAB SAC actor (no mshab imports).

Reimplements exactly the modules in mshab/agents/sac/modules.py so the released
checkpoint's ``actor.*`` weights load by name:

  encoder.cnns.{fetch_hand_depth,fetch_head_depth}   Conv(3->32->64->128->256, 3x3 s2 valid,
                                                     ReLU between convs, none after last) + flatten
  encoder.pixels_projections.{...}.projection        Linear(12544->50) + LayerNorm(50) + Tanh
  encoder.state_projection.projection                Linear(42->50)    + LayerNorm(50) + Tanh
  mlp                                                150 -> 256 -> 256 -> 256 -> 26 (ReLU between)

Deterministic action = tanh(mu) where (mu, log_std) = mlp(...).chunk(2).

Observation contract (verified against mshab @ e9ff3d2, ManiSkill mshab-branch @ 17121e3):
  pixels: dict with keys 'fetch_hand_depth', 'fetch_head_depth' (SORTED order — gym
      spaces.Dict sorts keys and the ModuleDicts were built from it), each
      (B, 3, 1, 128, 128) float32: RAW depth in millimeters, 3 stacked frames
      (oldest..newest), no normalization anywhere.
  state: (B, 42) float32 =
      qpos[3:] (12)              base x/y/theta joints stripped
    + qvel[3:] (12)
    + tcp_pose_wrt_base (7)      [p(3), q(4, wxyz)], base_link frame
    + obj_pose_wrt_base (7)      target object pose, base_link frame
    + goal_pos_wrt_base (3)      always zeros for the Pick subtask
    + is_grasped (1)             agent.is_grasping(target, max_angle=30)

Action (13, all in [-1, 1], unscaled by the env's controllers):
  [ arm delta pos x7 (+-0.1 rad) | gripper abs pos x1 (-0.01..0.05 m) |
    body delta pos x3 (head_pan, head_tilt, torso; +-0.1) | base x2 (fwd vel +-1, ang vel +-3.14) ]
  MS-HAB trains with stationary_head=True: the env wrapper zeroes dims -5 and -4
  (head_pan, head_tilt) AFTER the policy — callers must do the same.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PIXEL_KEYS = ("fetch_hand_depth", "fetch_head_depth")  # sorted, do not reorder
STATE_DIM = 42
ACTION_DIM = 13
HEAD_ACTION_DIMS = (-5, -4)  # zeroed when stationary_head (MS-HAB default)


class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class _SharedCNN(nn.Module):
    def __init__(self, in_channels=3, features=(32, 64, 128, 256)):
        super().__init__()
        layers = []
        ins = (in_channels,) + tuple(features[:-1])
        for i, (cin, cout) in enumerate(zip(ins, features)):
            layers.append(nn.Conv2d(cin, cout, 3, 2, padding=0))
            if i < len(features) - 1:
                layers.append(nn.ReLU())
        layers.append(_Flatten())
        self.layers = nn.Sequential(*layers)

    def forward(self, pixels):
        if pixels.dim() == 5:  # (B, stack, C, H, W) -> (B, stack*C, H, W)
            b, fs, c, h, w = pixels.shape
            pixels = pixels.view(b, fs * c, h, w).contiguous()
        return self.layers(pixels)


class _Projection(nn.Module):
    def __init__(self, in_dim, out_dim=50):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.projection(x)


class _Encoder(nn.Module):
    def __init__(self, cnn_flat_dim=12544, feat=50, state_dim=STATE_DIM):
        super().__init__()
        self.cnns = nn.ModuleDict({k: _SharedCNN() for k in PIXEL_KEYS})
        self.pixels_projections = nn.ModuleDict(
            {k: _Projection(cnn_flat_dim, feat) for k in PIXEL_KEYS}
        )
        self.state_projection = _Projection(state_dim, feat)

    def forward(self, pixels: dict, state: torch.Tensor):
        pix = torch.cat(
            [self.pixels_projections[k](self.cnns[k](pixels[k])) for k in PIXEL_KEYS],
            dim=1,
        )
        return torch.cat([pix, self.state_projection(state)], dim=1)


class SACActor(nn.Module):
    """Deterministic-inference actor; forward returns tanh(mu) in [-1, 1]."""

    def __init__(self, hidden=(256, 256, 256), action_dim=ACTION_DIM):
        super().__init__()
        self.encoder = _Encoder()
        dims = (150,) + tuple(hidden)
        layers = []
        for i, (din, dout) in enumerate(zip(dims, tuple(hidden) + (2 * action_dim,))):
            layers.append(nn.Linear(din, dout))
            if i < len(dims) - 1:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    @torch.no_grad()
    def forward(self, pixels: dict, state: torch.Tensor) -> torch.Tensor:
        x = self.encoder(pixels, state)
        mu, _log_std = self.mlp(x).chunk(2, dim=-1)
        return torch.tanh(mu)


def load_actor(ckpt_path: str, device: str = "cuda") -> SACActor:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    agent = ckpt["agent"] if "agent" in ckpt else ckpt
    actor_sd = {
        k[len("actor.") :]: v for k, v in agent.items() if k.startswith("actor.")
    }
    actor = SACActor()
    missing, unexpected = actor.load_state_dict(actor_sd, strict=True), None
    actor.to(device).eval()
    return actor
