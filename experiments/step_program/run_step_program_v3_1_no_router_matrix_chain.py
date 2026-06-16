#!/usr/bin/env python3
"""
StepProgram v3.1: no-router sequential matrix-chain experiment.

This is the cleanest test of the user's core idea:

  - fixed layer sequence L0 -> L1 -> L2 -> L3;
  - fixed step sequence S0 -> S1 -> S2;
  - fixed operation sequence inside every step;
  - no softmax family-choice;
  - no layer router;
  - no step-read router;
  - no identity primitive candidate;
  - all operations are always computed and always receive gradient;
  - specialization is measured through independent sigmoid gains and gradients.

Inside each MatrixChainUnit:

  h <- Norm(h + gain_small     * small_refine(h, ctx))
  h <- Norm(h + gain_diag      * diag_delta(h))
  h <- Norm(h + gain_low_rank  * low_rank(h))
  h <- Norm(h + gain_butterfly * butterfly(h))
  h <- Norm(h + gain_blockdiag * blockdiag(h))
  h <- Norm(h + gain_compare   * compare(h, ctx))
  h <- Norm(h + gain_norm      * normalize_delta(h))

The gains are independent sigmoids, not a softmax.  Multiple operations can be
active together.  This is closer to a trainable butterfly/product program than
to a router.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse dataset/frontend/report helpers from v3.  This file is a separate
# experiment, but does not duplicate data-loading code.
from run_step_program_v3_clean_sequential import (
    AudioEvidence,
    make_loaders,
    ensure_dir,
    amp_dtype,
    set_seed,
    entropy,
    write_json,
    write_jsonl,
    top_named,
)


OP_NAMES = ["small_refine", "diag_delta", "low_rank", "butterfly", "blockdiag", "compare", "normalize"]


@dataclass
class Aux:
    op_gains: torch.Tensor          # [B,L,Blocks,S,K,O]
    update_norms: torch.Tensor      # [B,L,Blocks,S,K,O]
    state_norms: torch.Tensor       # [B,L,Blocks,S,K]
    input_skip_gates: torch.Tensor  # [L]
    mean_skip_gates: torch.Tensor   # [L]
    slot_names: List[str]


class MatrixChainUnit(nn.Module):
    """Fixed operation chain with independent gains.

    There is no choice/softmax here.  Every operation runs in order.  Gains are
    independent and differentiable, so specialization is visible as per-address
    gain/grad/update_norm, not as selected branch.
    """
    def __init__(self, dim: int, blocks: int, rank: int = 24, groups: int = 8, dropout: float = 0.05, max_gain: float = 0.50):
        super().__init__()
        self.dim = int(dim)
        self.blocks = int(blocks)
        self.rank = int(rank)
        self.groups = int(groups) if dim % groups == 0 else 1
        self.group_size = dim // self.groups
        self.max_gain = float(max_gain)
        self.O = len(OP_NAMES)

        self.small = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.diag = nn.Parameter(torch.zeros(blocks, dim))
        self.low_a = nn.Linear(dim, rank, bias=False)
        self.low_b = nn.Linear(rank, dim, bias=False)
        self.bfly = nn.Parameter(torch.randn(blocks, max(1, dim // 2), 2, 2) * 0.025)
        self.block_w = nn.Parameter(torch.randn(blocks, self.groups, self.group_size, self.group_size) * 0.025)
        self.compare = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm_delta = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

        # Per-block, per-op base gains.  Start small-but-alive, not zero.  This
        # gives every op a gradient path while preventing early explosion.
        self.gain_logit = nn.Parameter(torch.full((blocks, self.O), -2.2))
        # Optional mild context modulation.  Still not a router: independent sigmoid gains.
        self.gain_context = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, self.O))
        nn.init.zeros_(self.gain_context[-1].weight)
        nn.init.zeros_(self.gain_context[-1].bias)

    def _butterfly_delta(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        if D % 2 != 0:
            x2 = F.pad(x, (0, 1))
        else:
            x2 = x
        D2 = x2.shape[-1]
        pairs = D2 // 2
        xp = x2.view(B, N, pairs, 2)
        W = self.bfly[:, :pairs].to(device=x.device, dtype=x.dtype)  # [N,pairs,2,2]
        y = torch.einsum("bnpi,npij->bnpj", xp, W).reshape(B, N, D2)
        return y[..., :D]

    def _blockdiag_delta(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        if self.groups == 1:
            W = self.block_w[:, 0].to(device=x.device, dtype=x.dtype)  # [N,D,D]
            return torch.einsum("bnd,ndf->bnf", x, W)
        xg = x.view(B, N, self.groups, self.group_size)
        W = self.block_w.to(device=x.device, dtype=x.dtype)
        return torch.einsum("bngd,ngdf->bngf", xg, W).reshape(B, N, D)

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = h.shape
        base_gain = self.gain_logit.to(device=h.device, dtype=h.dtype).view(1, N, self.O)
        ctx_gain = 0.25 * self.gain_context(torch.cat([h.detach(), context.detach()], dim=-1))
        gains = self.max_gain * torch.sigmoid(base_gain + ctx_gain)  # [B,N,O]

        gain_list: List[torch.Tensor] = []
        update_norms: List[torch.Tensor] = []
        state_norms: List[torch.Tensor] = []

        # 0 small_refine
        u = self.small(torch.cat([h, context], dim=-1))
        h = self.norm(h + gains[..., 0:1] * u)
        gain_list.append(gains[..., 0]); update_norms.append(u.float().norm(dim=-1)); state_norms.append(h.float().norm(dim=-1))

        # 1 diag_delta
        diag = torch.tanh(self.diag.to(device=h.device, dtype=h.dtype)).view(1, N, D)
        u = h * diag
        h = self.norm(h + gains[..., 1:2] * u)
        gain_list.append(gains[..., 1]); update_norms.append(u.float().norm(dim=-1)); state_norms.append(h.float().norm(dim=-1))

        # 2 low_rank
        u = self.low_b(self.low_a(h))
        h = self.norm(h + gains[..., 2:3] * u)
        gain_list.append(gains[..., 2]); update_norms.append(u.float().norm(dim=-1)); state_norms.append(h.float().norm(dim=-1))

        # 3 butterfly
        u = self._butterfly_delta(h)
        h = self.norm(h + gains[..., 3:4] * u)
        gain_list.append(gains[..., 3]); update_norms.append(u.float().norm(dim=-1)); state_norms.append(h.float().norm(dim=-1))

        # 4 blockdiag
        u = self._blockdiag_delta(h)
        h = self.norm(h + gains[..., 4:5] * u)
        gain_list.append(gains[..., 4]); update_norms.append(u.float().norm(dim=-1)); state_norms.append(h.float().norm(dim=-1))

        # 5 compare
        u = self.compare(torch.cat([h - context, h * context, context], dim=-1))
        h = self.norm(h + gains[..., 5:6] * u)
        gain_list.append(gains[..., 5]); update_norms.append(u.float().norm(dim=-1)); state_norms.append(h.float().norm(dim=-1))

        # 6 normalize_delta
        u = self.norm_delta(F.layer_norm(h, (D,)))
        h = self.norm(h + gains[..., 6:7] * u)
        gain_list.append(gains[..., 6]); update_norms.append(u.float().norm(dim=-1)); state_norms.append(h.float().norm(dim=-1))

        return h, torch.stack(gain_list, dim=-1), torch.stack(update_norms, dim=-1), torch.stack(state_norms, dim=-1)


class NoRouterMatrixChainNet(nn.Module):
    OP_NAMES = OP_NAMES

    def __init__(self, num_classes: int, dim: int, evidence_cells: int, layers: int, blocks: int, steps: int, substeps: int, sample_rate: int, n_mels: int, hop_length: int, rank: int, groups: int, dropout: float, max_gain: float):
        super().__init__()
        self.C, self.D = int(num_classes), int(dim)
        self.L, self.B, self.S, self.K = int(layers), int(blocks), int(steps), int(substeps)
        self.O = len(OP_NAMES)
        self.evidence = AudioEvidence(sample_rate, n_mels, hop_length, evidence_cells, dim)
        self.base_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.block_emb = nn.Parameter(torch.randn(self.B, dim) * 0.03)
        self.layer_emb = nn.Parameter(torch.randn(self.L, dim) * 0.03)
        self.step_emb = nn.Parameter(torch.randn(self.S, dim) * 0.03)
        self.task_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.03)

        self.input_skip_logit = nn.Parameter(torch.full((self.L,), -3.0))
        self.mean_skip_logit = nn.Parameter(torch.full((self.L,), -3.5))
        self.context_scale = nn.Parameter(torch.full((self.L, self.S), -2.5))
        self.input_q = nn.Linear(dim, dim, bias=False)
        self.input_k = nn.Linear(dim, dim, bias=False)
        self.input_v = nn.Linear(dim, dim, bias=False)

        self.units = nn.ModuleList([
            MatrixChainUnit(dim, blocks=blocks, rank=rank, groups=groups, dropout=dropout, max_gain=max_gain)
            for _ in range(self.L * self.S * self.K)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def _input_context(self, h: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        score = torch.einsum("bnd,bed->bne", self.input_q(h), self.input_k(evidence)) / math.sqrt(h.shape[-1])
        a = torch.softmax(score.float(), dim=-1).to(h.dtype)
        return torch.einsum("bne,bed->bnd", a, self.input_v(evidence))

    def forward(self, wav: torch.Tensor, return_aux: bool = True):
        B = wav.shape[0]
        device, dtype, D = wav.device, wav.dtype, self.D
        evidence = self.evidence(wav)
        base = self.base_proj(evidence.mean(dim=1))
        base_blocks = base[:, None, :] + self.block_emb.to(device=device, dtype=dtype).view(1, self.B, D)
        h = self.norm(base_blocks)
        task = self.task_emb.to(device=device, dtype=dtype).expand(B, self.B, D)

        gains_all: List[torch.Tensor] = []
        un_all: List[torch.Tensor] = []
        sn_all: List[torch.Tensor] = []
        slot_names: List[str] = []

        for l in range(self.L):
            prev = h
            prev_mean = prev.mean(dim=1, keepdim=True).expand(-1, self.B, -1)
            in_gate = torch.sigmoid(self.input_skip_logit[l]).to(device=device, dtype=dtype)
            mean_gate = torch.sigmoid(self.mean_skip_logit[l]).to(device=device, dtype=dtype)
            h = self.norm(prev + in_gate * base_blocks + mean_gate * prev_mean + self.layer_emb[l].to(device=device, dtype=dtype).view(1, 1, D))
            for s in range(self.S):
                inp_ctx = self._input_context(h + self.step_emb[s].to(device=device, dtype=dtype).view(1, 1, D), evidence)
                ctx_gate = torch.sigmoid(self.context_scale[l, s]).to(device=device, dtype=dtype)
                context = ctx_gate * inp_ctx + 0.10 * task
                for k in range(self.K):
                    unit = self.units[(l * self.S + s) * self.K + k]
                    h, gains, update_norms, state_norms = unit(h, context)
                    gains_all.append(gains)
                    un_all.append(update_norms)
                    sn_all.append(state_norms)
                for b in range(self.B):
                    slot_names.append(f"L{l}.B{b}.S{s}")

        final = self.norm(h.mean(dim=1))
        logits = self.head(final)
        if not return_aux:
            return logits, None
        gains_t = torch.stack(gains_all, dim=1).view(B, self.L, self.S, self.K, self.B, self.O).permute(0, 1, 4, 2, 3, 5).contiguous()
        un_t = torch.stack(un_all, dim=1).view(B, self.L, self.S, self.K, self.B, self.O).permute(0, 1, 4, 2, 3, 5).contiguous()
        sn_t = torch.stack(sn_all, dim=1).view(B, self.L, self.S, self.K, self.B, self.O).permute(0, 1, 4, 2, 3, 5).contiguous()
        return logits, Aux(
            op_gains=gains_t,
            update_norms=un_t,
            state_norms=sn_t,
            input_skip_gates=torch.sigmoid(self.input_skip_logit.detach()).cpu(),
            mean_skip_gates=torch.sigmoid(self.mean_skip_logit.detach()).cpu(),
            slot_names=slot_names,
        )


def retain_aux_grads(aux: Aux) -> None:
    for t in (aux.op_gains,):
        if t is not None and t.requires_grad:
            t.retain_grad()


def grad_x_gain(t: torch.Tensor, scale: float) -> Optional[torch.Tensor]:
    if t is None or t.grad is None:
        return None
    return (torch.nan_to_num((t.grad.detach().float() / max(1.0, scale)).abs()) * torch.nan_to_num(t.detach().float().abs())).mean(dim=0)


def build_grad_report(model: NoRouterMatrixChainNet, aux: Aux, scale: float) -> Dict:
    by: Dict[str, float] = {}
    top: List[Dict] = []
    gg = grad_x_gain(aux.op_gains, scale)
    if gg is not None:
        flat = gg.flatten()
        vals, idxs = torch.topk(flat, k=min(120, flat.numel()))
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            x = int(idx)
            o = x % model.O; x //= model.O
            k = x % model.K; x //= model.K
            s = x % model.S; x //= model.S
            b = x % model.B; x //= model.B
            l = x
            addr = f"L{l}.B{b}.S{s}.K{k}.op.{model.OP_NAMES[o]}"
            by[addr] = float(v)
            top.append({"type": "op", "address": addr, "grad_x_gain": float(v)})
    top.sort(key=lambda r: r["grad_x_gain"], reverse=True)
    return {"definition": "grad_x_gain=mean(abs(dloss/dgain)*gain), divided by GradScaler scale", "top": top[:120], "by_address": by}


def make_reports(model: NoRouterMatrixChainNet, aux: Aux, grad_report: Optional[Dict] = None):
    rows: List[Dict] = []
    levels = {"op": [], "utility": [], "layer": []}
    gains = aux.op_gains.detach().float().mean(0)       # [L,B,S,K,O]
    un = aux.update_norms.detach().float().mean(0)      # [L,B,S,K,O]
    sn = aux.state_norms.detach().float().mean(0)       # [L,B,S,K,O]
    by = (grad_report or {}).get("by_address", {})

    for l in range(model.L):
        levels["layer"].append({"address": f"L{l}", "input_skip_gate": float(aux.input_skip_gates[l]), "mean_skip_gate": float(aux.mean_skip_gates[l])})
        rows.append({"type": "layer", "address": f"L{l}.input_skip_gate", "value": float(aux.input_skip_gates[l])})
        rows.append({"type": "layer", "address": f"L{l}.mean_skip_gate", "value": float(aux.mean_skip_gates[l])})
        for b in range(model.B):
            for s in range(model.S):
                util = {
                    "address": f"L{l}.B{b}.S{s}",
                    "gain_mean": float(gains[l, b, s].mean()),
                    "gain_max": float(gains[l, b, s].max()),
                    "update_norm_mean": float(un[l, b, s].mean()),
                    "state_norm_mean": float(sn[l, b, s].mean()),
                    "gain_entropy_like": float(entropy((gains[l, b, s] / gains[l, b, s].sum(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_min(1e-8), -1).mean()),
                }
                levels["utility"].append(util)
                rows.append({"type": "utility", **util})
                for k in range(model.K):
                    top_ops = top_named(model.OP_NAMES, gains[l, b, s, k], 7)
                    levels["op"].append({"address": f"L{l}.B{b}.S{s}.K{k}", "top_gains": top_ops})
                    for o, name in enumerate(model.OP_NAMES):
                        addr = f"L{l}.B{b}.S{s}.K{k}.op.{name}"
                        rows.append({
                            "type": "op",
                            "address": addr,
                            "gain": float(gains[l, b, s, k, o]),
                            "prob": float(gains[l, b, s, k, o]),  # compatibility with query/analyzer tables
                            "update_norm": float(un[l, b, s, k, o]),
                            "state_norm": float(sn[l, b, s, k, o]),
                            "grad_x_gain": float(by.get(addr, 0.0)),
                        })
    summary = {
        "gain_mean": float(gains.mean()),
        "gain_max": float(gains.max()),
        "update_norm_mean": float(un.mean()),
        "state_norm_mean": float(sn.mean()),
        "input_skip_mean": float(aux.input_skip_gates.float().mean()),
        "mean_skip_mean": float(aux.mean_skip_gates.float().mean()),
    }
    return {"levels": levels, "summary": summary, "grad_report": grad_report}, rows


def aux_losses(aux: Aux, args):
    gains = aux.op_gains.float()
    gain_mean = gains.mean()
    update_norm = aux.update_norms.float().mean()
    min_gain = torch.tensor(float(args.min_gain_target), device=gains.device)
    return {
        "gain_mean": gain_mean,
        "update_norm_mean": update_norm,
        "gain_activity_loss": F.relu(min_gain - gain_mean).pow(2),
    }


def train_epoch(model, loader, opt, scaler, device, dtype, epoch, args):
    model.train()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    totals = {"loss": 0.0, "ce": 0.0, "correct": 0, "n": 0}
    sums = defaultdict(float)
    last_aux = None
    grad_rep = None
    for step, (wav, y) in enumerate(loader, 1):
        if args.max_train_batches and step > args.max_train_batches:
            break
        wav, y = wav.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, aux = model(wav, return_aux=True)
            ce = F.cross_entropy(logits.float(), y)
            losses = aux_losses(aux, args)
            loss = ce + float(args.lambda_gain_activity) * losses["gain_activity_loss"]
        do_grad = bool(args.grad_analytics_every and step % args.grad_analytics_every == 0)
        if do_grad:
            retain_aux_grads(aux)
        if not torch.isfinite(loss):
            print("NONFINITE_LOSS skip", flush=True)
            continue
        scale = float(scaler.get_scale()) if hasattr(scaler, "get_scale") else 1.0
        scaler.scale(loss).backward()
        if do_grad:
            grad_rep = build_grad_report(model, aux, scale)
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        bs = y.numel()
        totals["loss"] += float(loss.detach().cpu()) * bs
        totals["ce"] += float(ce.detach().cpu()) * bs
        totals["correct"] += int((logits.argmax(-1) == y).sum().detach().cpu())
        totals["n"] += bs
        for k, v in losses.items():
            sums[k] += float(v.detach().cpu()) * bs
        last_aux = Aux(
            op_gains=aux.op_gains.detach().cpu(),
            update_norms=aux.update_norms.detach().cpu(),
            state_norms=aux.state_norms.detach().cpu(),
            input_skip_gates=aux.input_skip_gates,
            mean_skip_gates=aux.mean_skip_gates,
            slot_names=aux.slot_names,
        )
        if args.log_every and step % args.log_every == 0:
            print(
                f"epoch {epoch:03d} step {step:05d} loss={totals['loss']/max(1,totals['n']):.4f} "
                f"ce={totals['ce']/max(1,totals['n']):.4f} acc={100*totals['correct']/max(1,totals['n']):.2f}% "
                f"gain={sums['gain_mean']/max(1,totals['n']):.3f} upd={sums['update_norm_mean']/max(1,totals['n']):.3f}",
                flush=True,
            )
    out = {"loss": totals["loss"] / max(1, totals["n"]), "ce": totals["ce"] / max(1, totals["n"]), "acc": totals["correct"] / max(1, totals["n"]), "n": totals["n"], "last_aux": last_aux, "grad_report": grad_rep}
    for k, v in sums.items():
        out[k] = v / max(1, totals["n"])
    return out


@torch.no_grad()
def evaluate(model, loader, device, dtype, args):
    model.eval()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    total_loss, correct, n = 0.0, 0, 0
    conf = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long)
    last_aux = None
    for step, (wav, y) in enumerate(loader, 1):
        if args.max_val_batches and step > args.max_val_batches:
            break
        wav, y = wav.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, aux = model(wav, return_aux=True)
            loss = F.cross_entropy(logits.float(), y)
        pred = logits.argmax(-1)
        bs = y.numel()
        total_loss += float(loss.cpu()) * bs
        correct += int((pred == y).sum().cpu())
        n += bs
        conf += torch.bincount((y.cpu() * args.num_classes + pred.cpu()), minlength=args.num_classes ** 2).view(args.num_classes, args.num_classes)
        last_aux = Aux(aux.op_gains.detach().cpu(), aux.update_norms.detach().cpu(), aux.state_norms.detach().cpu(), aux.input_skip_gates, aux.mean_skip_gates, aux.slot_names)
    return {"loss": total_loss / max(1, n), "acc": correct / max(1, n), "n": n, "confusion": conf.tolist(), "last_aux": last_aux}


def run(args):
    set_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    dtype = amp_dtype(args.amp)
    out_dir = ensure_dir(Path(args.out_dir))
    train_loader, val_loader, classes, train_counts, val_counts = make_loaders(args)
    args.num_classes = len(classes)
    model = NoRouterMatrixChainNet(
        len(classes), args.dim, args.evidence_cells, args.layers, args.blocks, args.steps, args.substeps,
        args.sample_rate, args.n_mels, args.hop_length, args.rank, args.groups, args.dropout, args.max_gain,
    ).to(device)
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"class counts train={train_counts} val={val_counts}", flush=True)
    print(f"NoRouterChainV3.1 params={sum(p.numel() for p in model.parameters())} L={args.layers} B={args.blocks} S={args.steps} K={args.substeps} O={len(OP_NAMES)} device={device} amp={args.amp}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)
    fields = ["epoch", "train_loss", "train_ce", "train_acc", "val_loss", "val_acc", "best_acc", "gain_mean", "update_norm_mean"]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    best, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, opt, scaler, device, dtype, epoch, args)
        va = evaluate(model, val_loader, device, dtype, args)
        if va["acc"] > best:
            best, best_epoch = va["acc"], epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best}, out_dir / "last.pt")
        report, rows = make_reports(model, va["last_aux"], tr.get("grad_report"))
        analysis = {"epoch": epoch, "train": {k: v for k, v in tr.items() if k not in ("last_aux", "grad_report")}, "val": {"loss": va["loss"], "acc": va["acc"], "n": va["n"], "confusion": va["confusion"]}, "best_acc": best, "best_epoch": best_epoch, "classes": classes, "train_counts": train_counts, "val_counts": val_counts, "program_report": report}
        write_json(out_dir / f"analysis_epoch_{epoch:03d}.json", analysis)
        write_jsonl(out_dir / f"events_epoch_{epoch:03d}.jsonl", [{"epoch": epoch, **r} for r in rows])
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_ce": tr["ce"], "train_acc": tr["acc"], "val_loss": va["loss"], "val_acc": va["acc"], "best_acc": best, "gain_mean": tr.get("gain_mean", 0.0), "update_norm_mean": tr.get("update_norm_mean", 0.0)}
        with (out_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch} gain={tr.get('gain_mean',0):.3f} upd={tr.get('update_norm_mean',0):.3f}", flush=True)
    write_json(out_dir / "final_report.json", {"best_acc": best, "best_epoch": best_epoch, "args": vars(args), "classes": classes, "train_counts": train_counts, "val_counts": val_counts})


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data/speechcommands")
    p.add_argument("--download", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--classes", default="yes,no,up,down,left,right,on,off,stop,go")
    p.add_argument("--train-limit", type=int, default=12000)
    p.add_argument("--val-limit", type=int, default=2000)
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--hop-length", type=int, default=160)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--evidence-cells", type=int, default=48)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--substeps", type=int, default=2)
    p.add_argument("--rank", type=int, default=24)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--max-gain", type=float, default=0.50)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.7)
    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="fp16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-gain-activity", type=float, default=0.0)
    p.add_argument("--min-gain-target", type=float, default=0.03)
    p.add_argument("--grad-analytics-every", type=int, default=50)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", default="./runs/step_program_v3_1_no_router_chain")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
