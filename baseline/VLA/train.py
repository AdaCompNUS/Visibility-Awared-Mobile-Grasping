"""SG-VLA baseline training (bf16, FSDP full-shard on GPUs 4-7).

Launch (from repo root; GPUs 4-7 ONLY):

  CUDA_VISIBLE_DEVICES=4,5,6,7 pixi run -e vla torchrun --standalone \
      --nproc_per_node=4 baseline/VLA/train.py --epochs 3

Single-GPU debug / smoke test:
  CUDA_VISIBLE_DEVICES=4 pixi run -e vla python baseline/VLA/train.py \
      --synthetic --debug-steps 3 --micro-batch 2

Throughput calibration (synthetic data, real config):
  CUDA_VISIBLE_DEVICES=4,5,6,7 pixi run -e vla torchrun --standalone \
      --nproc_per_node=4 baseline/VLA/train.py --synthetic --calib-steps 100

All paths are repo-relative (baseline/VLA/data, baseline/VLA/ckpts symlinks).
Hyperparameters not pinned by SG-VLA follow Prismatic/OpenVLA conventions —
every such default is documented in baseline/VLA/DECISIONS.md.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

VLA_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(VLA_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dataset import DATA_DIR, build_dataset, make_collate  # noqa: E402
from model.config import SGVLAConfig  # noqa: E402
from model.sgvla import SGVLA  # noqa: E402

CKPT_ROOT = VLA_ROOT / "ckpts"  # repo-relative symlink
ALLOWED_GPUS = {"4", "5", "6", "7"}


# ----------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=time.strftime("sgvla_%Y%m%d_%H%M%S"))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--ckpt-root", default=str(CKPT_ROOT))
    ap.add_argument("--synthetic", action="store_true", help="fabricated data (no h5 needed)")
    ap.add_argument("--synthetic-episodes", type=int, default=512)
    # optimization (SPEC-pinned unless noted in DECISIONS.md)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=4, help="per-GPU micro batch")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--frames-per-episode", type=int, default=30)
    # infra
    ap.add_argument("--num-workers", type=int, default=4, help="dataloader workers PER RANK (<=16 total)")
    ap.add_argument("--no-fsdp", action="store_true", help="single-process DDP-free debug path")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--save-every", type=int, default=1000, help="optimizer steps between checkpoints")
    ap.add_argument("--keep-last", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="ckpt dir with model.pt to warm-start from")
    ap.add_argument("--debug-steps", type=int, default=0, help="stop after N optimizer steps")
    ap.add_argument("--calib-steps", type=int, default=0,
                    help="throughput calibration: run N steps then report and exit")
    ap.add_argument("--allow-any-gpu", action="store_true")
    return ap.parse_args()


def guard_gpus(args):
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.allow_any_gpu:
        return
    if cvd is None or not set(filter(None, cvd.split(","))) <= ALLOWED_GPUS:
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES={cvd!r}: must be a subset of {{4,5,6,7}} "
            "(SPEC hard constraint). Pass --allow-any-gpu to override intentionally."
        )


# ----------------------------------------------------------------------
def lr_lambda_factory(warmup: int, total: int):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        p = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    return fn


def build_optimizer(model, args):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "embed" in n.lower() else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, args.beta2), eps=1e-8,
    )


def wrap_fsdp(model: SGVLA, local_rank: int) -> FSDP:
    from transformers.models.dinov2.modeling_dinov2 import Dinov2Layer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

    policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Dinov2Layer, Qwen2DecoderLayer, SiglipEncoderLayer},
    )
    return FSDP(
        model,
        auto_wrap_policy=policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32,  # keep norm-stat buffers exact
        ),
        device_id=local_rank,
        use_orig_params=True,
        limit_all_gathers=True,
    )


def save_checkpoint(model, cfg: SGVLAConfig, norm_stats: dict, run_dir: Path,
                    step: int, args, is_fsdp: bool, rank: int, keep_last: int):
    if is_fsdp:
        sd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, sd_cfg):
            sd = model.state_dict()
    else:
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
    if rank == 0:
        out = run_dir / f"step_{step:06d}"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(sd, out / "model.pt")
        cfg.to_json(out / "config.json")
        (out / "norm_stats.json").write_text(json.dumps(norm_stats, indent=2))
        (out / "trainer_state.json").write_text(json.dumps(
            {"step": step, "args": {k: str(v) for k, v in vars(args).items()}}, indent=2))
        latest = run_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out.name)
        # prune old checkpoints
        ckpts = sorted(run_dir.glob("step_*"))
        for old in ckpts[:-keep_last]:
            shutil.rmtree(old, ignore_errors=True)
        print(f"[ckpt] saved {out}", flush=True)


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    guard_gpus(args)

    distributed = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    assert args.global_batch % (world * args.micro_batch) == 0, \
        f"global_batch {args.global_batch} must divide by world*micro ({world}x{args.micro_batch})"
    accum = args.global_batch // (world * args.micro_batch)

    cfg = SGVLAConfig()
    run_dir = Path(args.ckpt_root) / args.run_name
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.to_json(run_dir / "config.json")

    # ---- data ----
    ds = build_dataset(cfg, synthetic=args.synthetic, data_dir=args.data_dir,
                       frames_per_episode=args.frames_per_episode,
                       n_synthetic_episodes=args.synthetic_episodes)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
    collate = make_collate(tokenizer, cfg.instr_max_len)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True,
                                 seed=args.seed) if distributed else None
    loader = DataLoader(
        ds, batch_size=args.micro_batch, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
        drop_last=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ---- normalization stats (rank 0 computes, others poll for the file) ----
    # Filesystem rendezvous instead of dist.barrier(): a lone early NCCL
    # collective deadlocked once while another job was spinning up on the same
    # GPUs (watchdog: SeqNum=1 ALLREDUCE timeout). The first NCCL collective
    # now happens inside FSDP where every rank reaches it at the same phase.
    stats_path = run_dir / "norm_stats.json"
    if rank == 0:
        norm_stats = ds.compute_norm_stats()
        tmp = stats_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(norm_stats, indent=2))
        tmp.rename(stats_path)  # atomic publish
    else:
        deadline = time.time() + 1800  # stats streaming over 9k episodes can take a while
        while not stats_path.exists():
            if time.time() > deadline:
                raise RuntimeError(f"rank {rank}: timed out waiting for {stats_path}")
            time.sleep(2)
    norm_stats = json.loads(stats_path.read_text())

    # ---- model ----
    t0 = time.time()
    model = SGVLA(cfg)
    model.set_norm_stats(norm_stats["state_mean"], norm_stats["state_std"],
                         norm_stats.get("robot_pos_mean"), norm_stats.get("robot_pos_std"))
    if args.resume:
        sd = torch.load(Path(args.resume) / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
        if rank == 0:
            print(f"[init] warm-started from {args.resume}")
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    if rank == 0:
        bd = model.param_breakdown()
        print("[init] params:", {k: f"{v/1e6:.1f}M" for k, v in bd.items()},
              f"| tokens/sample={model.tokens_per_sample} | load {time.time()-t0:.0f}s", flush=True)

    use_fsdp = distributed and not args.no_fsdp
    if use_fsdp:
        model = wrap_fsdp(model, local_rank)
    else:
        # single-GPU debug: pure bf16 (fp32 params + fp32 AdamW states measured
        # OOM on a 24 GB A5000 — that is why DDP is not the fallback here).
        model = model.cuda().to(torch.bfloat16)
        for b in model.buffers():
            b.data = b.data.float()  # keep normalization stats exact

    opt = build_optimizer(model, args)
    steps_per_epoch = len(ds) // args.global_batch
    total_steps = args.calib_steps or (steps_per_epoch * args.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.warmup_steps, total_steps))
    writer = SummaryWriter(run_dir / "tb") if rank == 0 else None
    if rank == 0:
        print(f"[init] world={world} micro={args.micro_batch} accum={accum} "
              f"global={args.global_batch} | {len(ds)} samples/epoch, "
              f"{steps_per_epoch} steps/epoch, {total_steps} total", flush=True)

    # ---- train loop ----
    model.train()
    step = 0
    t_step = time.time()
    step_times = []
    done = False
    for epoch in range(max(args.epochs, 1000 if args.calib_steps else args.epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        micro = 0
        for batch in loader:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=not use_fsdp)
            with amp:
                out = model(
                    rgb=batch["rgb"], depth_mm=batch["depth_mm"],
                    instr_ids=batch["instr_ids"], instr_mask=batch["instr_mask"],
                    state=batch["state"], action=batch["action"],
                    target_mask=batch["target_mask"], robot_pos=batch["robot_pos"],
                    robot_pos_valid=batch["robot_pos_valid"],
                )
            loss = out["loss_total"] / accum
            loss.backward()
            micro += 1
            if micro % accum != 0:
                continue

            if use_fsdp:
                grad_norm = model.clip_grad_norm_(args.grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            dt = time.time() - t_step
            t_step = time.time()
            if step > 10:
                step_times.append(dt)
            if rank == 0 and (step % args.log_every == 0 or step <= 5):
                mem = torch.cuda.max_memory_allocated() / 2**30
                msg = " ".join(f"{k[5:]}={out[k].item():.4f}" for k in sorted(out) if k.startswith("loss_"))
                print(f"[e{epoch} s{step}/{total_steps}] {msg} "
                      f"gnorm={float(grad_norm):.2f} lr={sched.get_last_lr()[0]:.2e} "
                      f"{dt:.2f}s/step mem={mem:.1f}GiB", flush=True)
                if writer:
                    for k in out:
                        if k.startswith("loss_"):
                            writer.add_scalar(f"train/{k}", out[k].item(), step)
                    writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                    writer.add_scalar("train/grad_norm", float(grad_norm), step)
                    writer.add_scalar("train/sec_per_step", dt, step)
                    writer.add_scalar("train/mem_gib", mem, step)

            if args.save_every and step % args.save_every == 0 and not args.calib_steps:
                save_checkpoint(model, cfg, norm_stats, run_dir, step, args,
                                use_fsdp, rank, args.keep_last)
            if (args.debug_steps and step >= args.debug_steps) or \
               (args.calib_steps and step >= args.calib_steps):
                done = True
                break
        if done or args.debug_steps:
            break
        if not args.calib_steps:
            save_checkpoint(model, cfg, norm_stats, run_dir, step, args,
                            use_fsdp, rank, args.keep_last)
    if not (args.calib_steps or args.debug_steps):
        save_checkpoint(model, cfg, norm_stats, run_dir, step, args, use_fsdp, rank, args.keep_last)

    # ---- calibration report ----
    if rank == 0 and step_times:
        s = sorted(step_times)
        med = s[len(s) // 2]
        mem = torch.cuda.max_memory_allocated() / 2**30
        tokens = cfg.instr_max_len + cfg.history * cfg.n_views * cfg.vision_grid ** 2 + 3
        print(f"\n[calib] steps={len(step_times)} median={med:.2f}s/step "
              f"mean={sum(step_times)/len(step_times):.2f}s "
              f"| global_batch={args.global_batch} world={world} "
              f"| VRAM(rank0 max)={mem:.1f}GiB | tokens/sample={tokens}",
              flush=True)
    if writer:
        writer.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
