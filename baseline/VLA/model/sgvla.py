"""SG-VLA-style model (~1.24B): DINOv2-L + SigLIP-SO400M -> projector ->
Qwen2.5-0.5B -> continuous 13-d action head + 5 auxiliary decoders.

Token sequence fed to the (causal) LLM, in order:

  [instruction (32, right-padded)] [vision t-3 head|hand ... t head|hand
   (4*2*256 = 2048)] [AUX query (1)] [proprio (1)] [ACT query (1)]

Instruction first so vision tokens can attend to it (causal attention); the
AUX query sits BEFORE the proprio token so the auxiliary decoders must answer
from vision+language alone (qpos / is_grasped / target pose would otherwise be
copied from the proprio input); the ACT query is last and attends everything.
Total sequence = 2083 tokens.

Depth (SPEC formula 1 - tanh(d_mm/1000)) enters as an auxiliary channel:
each 128x128 depth map is cut into the same 16x16 grid as the vision tokens
(8x8 px per patch -> 64 values), linearly embedded to the fused vision width
and ADDED to the corresponding token before the projector.

The target-mask decoder reads the LLM's output hidden states at the
current-frame HEAD-camera token positions (16x16 x 896) and upsamples to a
128x128 logit map.

Normalization statistics (proprio state mean/std, optional robot-position
mean/std) are registered buffers -> they live inside every checkpoint.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SGVLAConfig
from .vision import DualVisionBackbone


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class MaskDecoder(nn.Module):
    """(B, 256, llm_dim) head-cam tokens -> (B, 128, 128) mask logits."""

    def __init__(self, llm_dim: int, grid: int, out_size: int, base: int = 256):
        super().__init__()
        self.grid = grid
        assert out_size % grid == 0
        n_up = int(math.log2(out_size // grid))  # 16 -> 128: 3 doublings
        chans = [base] + [max(base // (2 ** (i + 1)), 32) for i in range(n_up)]
        self.inp = nn.Conv2d(llm_dim, base, 3, padding=1)
        ups = []
        for i in range(n_up):
            ups += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1),
                nn.GELU(),
            ]
        self.ups = nn.Sequential(*ups)
        self.out = nn.Conv2d(chans[-1], 1, 3, padding=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, c, self.grid, self.grid)
        x = F.gelu(self.inp(x))
        x = self.ups(x)
        return self.out(x).squeeze(1)


class SGVLA(nn.Module):
    def __init__(self, cfg: SGVLAConfig):
        super().__init__()
        self.cfg = cfg

        # ---- pretrained backbones ----
        self.vision = DualVisionBackbone(cfg.dino_model, cfg.siglip_model, cfg.image_size)
        from transformers import AutoModel

        self.llm = AutoModel.from_pretrained(cfg.llm_model)  # Qwen2Model (no LM head)
        llm_dim = self.llm.config.hidden_size  # 896
        fused = self.vision.embed_dim          # 2176
        grid = cfg.vision_grid
        self.llm_dim = llm_dim

        # ---- from-scratch modules ----
        patch_px = cfg.source_image_size // grid  # 8
        self.depth_embed = nn.Linear(patch_px * patch_px, fused)
        self.projector = _mlp(fused, cfg.projector_hidden, llm_dim)
        self.state_encoder = _mlp(cfg.state_dim, llm_dim, llm_dim)
        self.frame_embed = nn.Embedding(cfg.history, llm_dim)
        self.view_embed = nn.Embedding(cfg.n_views, llm_dim)
        self.aux_query = nn.Parameter(torch.zeros(1, 1, llm_dim))
        self.act_query = nn.Parameter(torch.zeros(1, 1, llm_dim))
        nn.init.normal_(self.aux_query, std=0.02)
        nn.init.normal_(self.act_query, std=0.02)

        self.action_head = _mlp(llm_dim, cfg.action_head_hidden, cfg.action_dim)
        self.mask_decoder = MaskDecoder(llm_dim, grid, cfg.mask_size, cfg.mask_decoder_base)
        h = cfg.aux_head_hidden
        self.robot_pos_head = _mlp(llm_dim, h, 3)
        self.qpos_head = _mlp(llm_dim, h, 12)
        self.grasped_head = _mlp(llm_dim, h, 1)
        self.target_pose_head = _mlp(llm_dim, h, 7)

        # ---- normalization stats (filled by train.py, saved in ckpt) ----
        self.register_buffer("state_mean", torch.zeros(cfg.state_dim))
        self.register_buffer("state_std", torch.ones(cfg.state_dim))
        self.register_buffer("robot_pos_mean", torch.zeros(3))
        self.register_buffer("robot_pos_std", torch.ones(3))

    # ------------------------------------------------------------------
    def set_norm_stats(self, state_mean, state_std, robot_pos_mean=None, robot_pos_std=None):
        def _t(x):
            return torch.as_tensor(x, dtype=torch.float32)

        self.state_mean.copy_(_t(state_mean))
        self.state_std.copy_(_t(state_std).clamp_min(1e-2))
        if robot_pos_mean is not None:
            self.robot_pos_mean.copy_(_t(robot_pos_mean))
            self.robot_pos_std.copy_(_t(robot_pos_std).clamp_min(1e-2))

    def gradient_checkpointing_enable(self):
        self.vision.gradient_checkpointing_enable()
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ------------------------------------------------------------------
    def _sequence_layout(self):
        cfg = self.cfg
        n_img = cfg.history * cfg.n_views
        n_vis = n_img * self.vision.num_tokens
        i0 = cfg.instr_max_len
        # head cam of the newest frame: frame index history-1, view index 0
        m0 = i0 + ((cfg.history - 1) * cfg.n_views + 0) * self.vision.num_tokens
        return {
            "instr": (0, i0),
            "vision": (i0, i0 + n_vis),
            "head_now": (m0, m0 + self.vision.num_tokens),
            "aux": i0 + n_vis,
            "proprio": i0 + n_vis + 1,
            "act": i0 + n_vis + 2,
            "total": i0 + n_vis + 3,
        }

    @property
    def tokens_per_sample(self) -> int:
        return self._sequence_layout()["total"]

    # ------------------------------------------------------------------
    def encode(
        self,
        rgb: torch.Tensor,        # (B, T, V, H, W, 3) uint8 or float[0,1]
        depth_mm: torch.Tensor,   # (B, T, V, H, W) float32, RAW millimeters
        instr_ids: torch.Tensor,  # (B, L<=32) int64
        instr_mask: torch.Tensor, # (B, L) bool/int
        state: torch.Tensor,      # (B, 42) float32, RAW mshab state
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        lay = self._sequence_layout()
        B, T, V, H, W, _ = rgb.shape
        assert T == cfg.history and V == cfg.n_views
        dev = self.state_mean.device
        dt = next(self.projector.parameters()).dtype

        # ---- vision ----
        if rgb.dtype == torch.uint8:
            rgb01 = rgb.to(dt) / 255.0
        else:
            rgb01 = rgb.to(dt)
        rgb01 = rgb01.permute(0, 1, 2, 5, 3, 4).reshape(B * T * V, 3, H, W)
        feats = self.vision(rgb01)  # (B*T*V, 256, 2176)

        # ---- depth as auxiliary channel on the aligned token grid ----
        d = 1.0 - torch.tanh(depth_mm.to(dt) / 1000.0)          # SPEC formula
        g, p = cfg.vision_grid, cfg.source_image_size // cfg.vision_grid
        d = d.reshape(B * T * V, g, p, g, p).permute(0, 1, 3, 2, 4).reshape(B * T * V, g * g, p * p)
        feats = feats + self.depth_embed(d)

        vis = self.projector(feats).reshape(B, T, V, self.vision.num_tokens, -1)
        vis = vis + self.frame_embed.weight.view(1, T, 1, 1, -1) + self.view_embed.weight.view(1, 1, V, 1, -1)
        vis = vis.reshape(B, T * V * self.vision.num_tokens, -1)

        # ---- text ----
        L = instr_ids.shape[1]
        assert L <= cfg.instr_max_len
        if L < cfg.instr_max_len:
            pad = cfg.instr_max_len - L
            instr_ids = F.pad(instr_ids, (0, pad), value=0)
            instr_mask = F.pad(instr_mask.to(torch.bool), (0, pad), value=False)
        txt = self.llm.get_input_embeddings()(instr_ids).to(dt)

        # ---- proprio + queries ----
        s_norm = (state.to(dt) - self.state_mean.to(dt)) / self.state_std.to(dt)
        prop = self.state_encoder(s_norm).unsqueeze(1)
        aux_q = self.aux_query.to(dt).expand(B, -1, -1)
        act_q = self.act_query.to(dt).expand(B, -1, -1)

        seq = torch.cat([txt, vis, aux_q, prop, act_q], dim=1)
        attn = torch.ones(B, lay["total"], dtype=torch.bool, device=seq.device)
        attn[:, : cfg.instr_max_len] = instr_mask.to(torch.bool)

        h = self.llm(inputs_embeds=seq, attention_mask=attn).last_hidden_state

        m0, m1 = lay["head_now"]
        return {
            "action": self.action_head(h[:, lay["act"]]),
            "mask_logits": self.mask_decoder(h[:, m0:m1]),
            "robot_pos": self.robot_pos_head(h[:, lay["aux"]]),
            "qpos": self.qpos_head(h[:, lay["aux"]]),
            "grasped_logit": self.grasped_head(h[:, lay["aux"]]).squeeze(-1),
            "target_pose": self.target_pose_head(h[:, lay["aux"]]),
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        rgb, depth_mm, instr_ids, instr_mask, state,
        action: Optional[torch.Tensor] = None,        # (B, 13) target in [-1,1]
        target_mask: Optional[torch.Tensor] = None,   # (B, 128, 128) float {0,1}
        robot_pos: Optional[torch.Tensor] = None,     # (B, 3) global base x/y/theta
        robot_pos_valid: Optional[torch.Tensor] = None,  # (B,) bool
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        out = self.encode(rgb, depth_mm, instr_ids, instr_mask, state)
        if action is None:
            return out

        f32 = torch.float32
        losses = {}
        losses["action"] = F.l1_loss(out["action"].to(f32), action.to(f32))
        if target_mask is not None:
            losses["mask"] = F.binary_cross_entropy_with_logits(
                out["mask_logits"].to(f32), target_mask.to(f32))
        # aux regression targets come normalized with the SAME stats as the input state
        sm, ss = self.state_mean, self.state_std
        q0, q1 = cfg.qpos_slice
        t0, t1 = cfg.target_pose_slice
        s = state.to(f32)
        losses["qpos"] = F.l1_loss(out["qpos"].to(f32), (s[:, q0:q1] - sm[q0:q1]) / ss[q0:q1])
        losses["target_pose"] = F.l1_loss(out["target_pose"].to(f32), (s[:, t0:t1] - sm[t0:t1]) / ss[t0:t1])
        losses["grasped"] = F.binary_cross_entropy_with_logits(
            out["grasped_logit"].to(f32), s[:, cfg.grasped_index].clamp(0, 1))
        if robot_pos is not None and robot_pos_valid is not None and robot_pos_valid.any():
            v = robot_pos_valid.to(torch.bool)
            tgt = (robot_pos.to(f32)[v] - self.robot_pos_mean) / self.robot_pos_std
            losses["robot_pos"] = F.l1_loss(out["robot_pos"].to(f32)[v], tgt)

        weights = {
            "action": cfg.w_action, "mask": cfg.w_mask, "robot_pos": cfg.w_robot_pos,
            "qpos": cfg.w_qpos, "grasped": cfg.w_grasped, "target_pose": cfg.w_target_pose,
        }
        losses["total"] = sum(weights[k] * v for k, v in losses.items())
        out.update({f"loss_{k}": v for k, v in losses.items()})
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action(self, rgb, depth_mm, instr_ids, instr_mask, state) -> torch.Tensor:
        out = self.encode(rgb, depth_mm, instr_ids, instr_mask, state)
        return out["action"].float().clamp(-1.0, 1.0)

    def param_breakdown(self) -> Dict[str, int]:
        def cnt(m):
            return sum(p.numel() for p in m.parameters())

        new = (cnt(self.projector) + cnt(self.depth_embed) + cnt(self.state_encoder)
               + cnt(self.frame_embed) + cnt(self.view_embed) + 2 * self.llm_dim
               + cnt(self.action_head) + cnt(self.mask_decoder) + cnt(self.robot_pos_head)
               + cnt(self.qpos_head) + cnt(self.grasped_head) + cnt(self.target_pose_head))
        return {
            "dinov2": cnt(self.vision.dino),
            "siglip": cnt(self.vision.siglip),
            "llm": cnt(self.llm),
            "new_modules": new,
            "total": cnt(self),
        }
