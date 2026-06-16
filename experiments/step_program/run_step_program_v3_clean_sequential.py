#!/usr/bin/env python3
"""
StepProgram v3: clean sequential matrix-program core.

Principle:
  - no layer-route soup;
  - no free step-read router;
  - no identity primitive candidate;
  - strict layer sequence: L0 -> L1 -> L2 -> L3;
  - strict step sequence inside every layer/block: S0 -> S1 -> S2;
  - each step contains K MatrixFamilyUnit substeps;
  - identity is only the residual skip: h <- Norm(h + write_gate * F(h));
  - output head reads only the final layer, so it cannot bypass the program.

The learnable choice is inside each MatrixFamilyUnit:

  F(h, ctx) = sum_i softmax(gate)_i * Family_i(h, ctx)

Families:
  small_refine, diag_delta, low_rank, butterfly, blockdiag, compare, normalize

Everything is differentiable.  The goal is to test the clean butterfly-like
idea before adding projected top-k, attention, growth, or plateau controllers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import torchaudio
except Exception:
    torchaudio = None


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def amp_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    q = torch.nan_to_num(p.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-8)
    return -(q * q.log()).sum(dim=dim)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def top_named(names: Sequence[str], weights: torch.Tensor, k: int = 5) -> List[Dict]:
    w = torch.nan_to_num(weights.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if w.numel() == 0:
        return []
    vals, idxs = torch.topk(w, k=min(int(k), int(w.numel())))
    return [{"name": names[int(i)], "value": float(v)} for v, i in zip(vals.tolist(), idxs.tolist())]


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


class SyntheticSpeechLike(Dataset):
    def __init__(self, n: int, num_classes: int, sample_rate: int = 16000, seconds: float = 1.0):
        self.n = int(n)
        self.num_classes = int(num_classes)
        self.sample_rate = int(sample_rate)
        self.length = int(sample_rate * seconds)
        self.t = torch.linspace(0, seconds, self.length)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        y = int(idx % self.num_classes)
        freq = 180.0 + 55.0 * y
        phase = 0.07 * (idx % 23)
        wav = torch.sin(2.0 * math.pi * freq * self.t + phase) + 0.04 * torch.randn_like(self.t)
        return wav.unsqueeze(0), y


class SpeechCommandsBalanced(Dataset):
    """Balanced filtered SpeechCommands subset.

    Older experiments used the first N matching files, which can create class/order
    bias.  v3 uses balanced per-class sampling when a limit is given.
    """
    def __init__(self, root: str, subset: str, classes: Sequence[str], limit: int = 0, download: bool = False):
        if torchaudio is None:
            raise RuntimeError("torchaudio is not available; use --synthetic for smoke tests")
        self.classes = list(classes)
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(root=root, subset=subset, download=download)
        per_class_limit = None
        if limit and limit > 0:
            per_class_limit = max(1, math.ceil(limit / max(1, len(self.classes))))
        buckets: Dict[str, List[int]] = {c: [] for c in self.classes}
        for i in range(len(self.ds)):
            try:
                label = Path(self.ds._walker[i]).parent.name
            except Exception:
                label = str(self.ds[i][2])
            if label not in self.class_to_id:
                continue
            if per_class_limit is not None and len(buckets[label]) >= per_class_limit:
                continue
            buckets[label].append(i)
            if per_class_limit is not None and all(len(buckets[c]) >= per_class_limit for c in self.classes):
                break
        keep: List[int] = []
        # Interleave classes for stable early batches.
        max_len = max((len(v) for v in buckets.values()), default=0)
        for j in range(max_len):
            for c in self.classes:
                if j < len(buckets[c]):
                    keep.append(buckets[c][j])
                    if limit and len(keep) >= limit:
                        break
            if limit and len(keep) >= limit:
                break
        self.keep = keep
        self.counts = {c: len(buckets[c]) for c in self.classes}

    def __len__(self) -> int:
        return len(self.keep)

    def __getitem__(self, j: int):
        wav, sr, label, *_ = self.ds[self.keep[j]]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.float(), self.class_to_id[str(label)]


def collate_wavs(batch, sample_rate: int, seconds: float):
    target = int(sample_rate * seconds)
    wavs, ys = [], []
    for wav, y in batch:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[-1] < target:
            wav = F.pad(wav, (0, target - wav.shape[-1]))
        elif wav.shape[-1] > target:
            wav = wav[..., :target]
        wavs.append(wav)
        ys.append(int(y))
    return torch.stack(wavs, dim=0), torch.tensor(ys, dtype=torch.long)


def make_loaders(args):
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.synthetic:
        train_ds = SyntheticSpeechLike(args.train_limit or 512, len(classes), args.sample_rate, args.seconds)
        val_ds = SyntheticSpeechLike(args.val_limit or 256, len(classes), args.sample_rate, args.seconds)
        train_counts = {c: (args.train_limit or 512) // len(classes) for c in classes}
        val_counts = {c: (args.val_limit or 256) // len(classes) for c in classes}
    else:
        train_ds = SpeechCommandsBalanced(args.data_root, "training", classes, args.train_limit, args.download)
        val_ds = SpeechCommandsBalanced(args.data_root, "validation", classes, args.val_limit, args.download)
        train_counts = getattr(train_ds, "counts", {})
        val_counts = getattr(val_ds, "counts", {})
    collate = lambda b: collate_wavs(b, args.sample_rate, args.seconds)
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory, collate_fn=collate)
    val = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.pin_memory, collate_fn=collate)
    return train, val, classes, train_counts, val_counts


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


@dataclass
class Aux:
    family_gates: torch.Tensor       # [B,L,N,S,K,F]
    write_gates: torch.Tensor        # [B,L,N,S,K]
    update_norms: torch.Tensor       # [B,L,N,S,K]
    slot_states: torch.Tensor        # [B,slots,D]
    slot_names: List[str]
    input_skip_gates: torch.Tensor   # [L]
    mean_skip_gates: torch.Tensor    # [L]


class AudioEvidence(nn.Module):
    def __init__(self, sample_rate: int, n_mels: int, hop_length: int, evidence_cells: int, dim: int):
        super().__init__()
        self.evidence_cells = int(evidence_cells)
        if torchaudio is not None:
            self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=400, hop_length=hop_length, n_mels=n_mels, power=2.0)
        else:
            self.mel = None
        self.scalar_proj = nn.Sequential(nn.LayerNorm(1), nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.randn(evidence_cells, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wav.device.type, enabled=False):
            wav32 = wav.float()
            if self.mel is not None:
                x = self.mel(wav32.squeeze(1)).float().clamp_min(1e-5).log()
            else:
                x = wav32.squeeze(1).unfold(-1, 320, 160).abs().mean(dim=-1).unsqueeze(1).repeat(1, 64, 1)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = (x - x.mean(dim=(-2, -1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            grid = F.adaptive_avg_pool2d(x.unsqueeze(1), (6, 6)).squeeze(1).flatten(1)
            time3 = F.adaptive_avg_pool2d(x.mean(dim=1, keepdim=True), (1, 3)).flatten(1)
            freq4 = F.adaptive_avg_pool2d(x.mean(dim=2, keepdim=True), (4, 1)).flatten(1)
            stats = torch.stack([x.mean(dim=(-2, -1)), x.std(dim=(-2, -1)), x.amax(dim=(-2, -1)), x.amin(dim=(-2, -1))], dim=1)
            delta_time = time3[:, 1:] - time3[:, :-1]
            feats = torch.cat([grid, time3, freq4, stats, delta_time], dim=1)
            if feats.shape[1] < self.evidence_cells:
                feats = feats.repeat(1, math.ceil(self.evidence_cells / feats.shape[1]))
            feats = feats[:, : self.evidence_cells]
        return self.norm(self.scalar_proj(feats.unsqueeze(-1)) + self.pos.view(1, self.evidence_cells, -1))


class MatrixFamilyUnit(nn.Module):
    FAMILY_NAMES = ["small_refine", "diag_delta", "low_rank", "butterfly", "blockdiag", "compare", "normalize"]

    def __init__(self, dim: int, rank: int = 24, groups: int = 8, dropout: float = 0.05):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.groups = int(groups)
        if self.dim % self.groups != 0:
            self.groups = 1
        self.group_size = self.dim // self.groups
        self.F = len(self.FAMILY_NAMES)

        self.gate_net = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, self.F))
        # Mild phase prior: start from useful transforms, not identity.  This is a buffer, not a learned shortcut.
        self.register_buffer("family_prior", torch.tensor([0.4, -0.2, 0.1, 0.1, 0.1, 0.2, 0.2], dtype=torch.float32))

        self.small = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.diag = nn.Parameter(torch.zeros(dim))
        self.low_a = nn.Linear(dim, rank, bias=False)
        self.low_b = nn.Linear(rank, dim, bias=False)

        pairs = max(1, dim // 2)
        self.bfly = nn.Parameter(torch.randn(pairs, 2, 2) * 0.03)
        self.block_w = nn.Parameter(torch.randn(self.groups, self.group_size, self.group_size) * 0.03)

        self.compare = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm_op = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.write_net = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.out_norm = nn.LayerNorm(dim)

    def _butterfly_delta(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        if D % 2 != 0:
            x2 = F.pad(x, (0, 1))
        else:
            x2 = x
        D2 = x2.shape[-1]
        pairs = D2 // 2
        xp = x2.view(B, N, pairs, 2)
        W = self.bfly[:pairs].to(dtype=x.dtype, device=x.device)
        y = torch.einsum("bnpi,pij->bnpj", xp, W).reshape(B, N, D2)
        return y[..., :D]

    def _blockdiag_delta(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        if self.groups == 1:
            W = self.block_w[0].to(dtype=x.dtype, device=x.device)
            return torch.einsum("bnd,df->bnf", x, W)
        xg = x.view(B, N, self.groups, self.group_size)
        W = self.block_w.to(dtype=x.dtype, device=x.device)
        y = torch.einsum("bngd,gdf->bngf", xg, W).reshape(B, N, D)
        return y

    def forward(self, x: torch.Tensor, context: torch.Tensor, temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        D = x.shape[-1]
        gate_logits = self.gate_net(torch.cat([x, context, x * context], dim=-1))
        gate_logits = gate_logits + self.family_prior.to(device=x.device, dtype=gate_logits.dtype).view(1, 1, self.F)
        gates = torch.softmax((gate_logits / max(1e-4, float(temp))).float(), dim=-1).to(x.dtype)

        small = self.small(torch.cat([x, context], dim=-1))
        diag = x * torch.tanh(self.diag.to(device=x.device, dtype=x.dtype)).view(1, 1, D)
        low = self.low_b(self.low_a(x))
        bfly = self._butterfly_delta(x)
        block = self._blockdiag_delta(x)
        comp = self.compare(torch.cat([x - context, x * context, context], dim=-1))
        normed = self.norm_op(F.layer_norm(x, (D,)))
        cands = torch.stack([small, diag, low, bfly, block, comp, normed], dim=2)  # [B,N,F,D]
        update = torch.einsum("bnf,bnfd->bnd", gates, cands)
        write_gate = torch.sigmoid(self.write_net(torch.cat([x, update, context], dim=-1))).squeeze(-1)
        y = self.out_norm(x + write_gate.unsqueeze(-1) * update)
        update_norm = update.float().norm(dim=-1)
        return y, gates, write_gate, update_norm


class CleanSequentialMatrixProgram(nn.Module):
    FAMILY_NAMES = MatrixFamilyUnit.FAMILY_NAMES

    def __init__(self, num_classes: int, dim: int, evidence_cells: int, layers: int, blocks: int, steps: int, substeps: int, sample_rate: int, n_mels: int, hop_length: int, rank: int, groups: int, dropout: float):
        super().__init__()
        self.C = int(num_classes)
        self.D = int(dim)
        self.E = int(evidence_cells)
        self.L = int(layers)
        self.B = int(blocks)
        self.S = int(steps)
        self.K = int(substeps)
        self.F = len(self.FAMILY_NAMES)

        self.evidence = AudioEvidence(sample_rate, n_mels, hop_length, evidence_cells, dim)
        self.base_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.block_emb = nn.Parameter(torch.randn(self.B, dim) * 0.03)
        self.layer_emb = nn.Parameter(torch.randn(self.L, dim) * 0.03)
        self.step_emb = nn.Parameter(torch.randn(self.S, dim) * 0.03)
        self.task_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.03)

        # Small learnable sequential side paths.  They are not routers.
        self.input_skip_logit = nn.Parameter(torch.full((self.L,), -3.0))
        self.mean_skip_logit = nn.Parameter(torch.full((self.L,), -3.5))

        self.input_q = nn.Linear(dim, dim, bias=False)
        self.input_k = nn.Linear(dim, dim, bias=False)
        self.input_v = nn.Linear(dim, dim, bias=False)
        self.context_scale = nn.Parameter(torch.full((self.L, self.S), -2.5))

        self.units = nn.ModuleList([
            MatrixFamilyUnit(dim, rank=rank, groups=groups, dropout=dropout)
            for _ in range(self.L * self.S * self.K)
        ])
        self.layer_norm = nn.LayerNorm(dim)
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def _input_context(self, h: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        score = torch.einsum("bnd,bed->bne", self.input_q(h), self.input_k(evidence)) / math.sqrt(h.shape[-1])
        a = torch.softmax(score.float(), dim=-1).to(h.dtype)
        return torch.einsum("bne,bed->bnd", a, self.input_v(evidence))

    def forward(self, wav: torch.Tensor, family_temp: float = 1.0, return_aux: bool = True):
        B = wav.shape[0]
        device = wav.device
        dtype = wav.dtype
        D = self.D
        evidence = self.evidence(wav)
        base = self.base_proj(evidence.mean(dim=1))
        base_blocks = base[:, None, :] + self.block_emb.to(device=device, dtype=dtype).view(1, self.B, D)
        h = self.layer_norm(base_blocks)
        task = self.task_emb.to(device=device, dtype=dtype).expand(B, self.B, D)

        family_all: List[torch.Tensor] = []
        write_all: List[torch.Tensor] = []
        update_norm_all: List[torch.Tensor] = []
        slot_states: List[torch.Tensor] = []
        slot_names: List[str] = []

        for l in range(self.L):
            prev = h
            prev_mean = prev.mean(dim=1, keepdim=True).expand(-1, self.B, -1)
            input_gate = torch.sigmoid(self.input_skip_logit[l]).to(device=device, dtype=dtype)
            mean_gate = torch.sigmoid(self.mean_skip_logit[l]).to(device=device, dtype=dtype)
            h = self.layer_norm(
                prev
                + input_gate * base_blocks
                + mean_gate * prev_mean
                + self.layer_emb[l].to(device=device, dtype=dtype).view(1, 1, D)
            )
            for s in range(self.S):
                inp_ctx = self._input_context(h + self.step_emb[s].to(device=device, dtype=dtype).view(1, 1, D), evidence)
                ctx_gate = torch.sigmoid(self.context_scale[l, s]).to(device=device, dtype=dtype)
                context = ctx_gate * inp_ctx + 0.10 * task
                # Strict substep chain: K matrix-family units in sequence.
                for k in range(self.K):
                    idx = (l * self.S + s) * self.K + k
                    h, fg, wg, un = self.units[idx](h, context=context, temp=family_temp)
                    family_all.append(fg)
                    write_all.append(wg)
                    update_norm_all.append(un)
                slot_states.append(h)
                for b in range(self.B):
                    slot_names.append(f"L{l}.B{b}.S{s}")

        final = self.final_norm(h.mean(dim=1))
        logits = self.head(final)
        if not return_aux:
            return logits, None

        fam = torch.stack(family_all, dim=1).view(B, self.L, self.S, self.K, self.B, self.F).permute(0, 1, 4, 2, 3, 5).contiguous()
        wr = torch.stack(write_all, dim=1).view(B, self.L, self.S, self.K, self.B).permute(0, 1, 4, 2, 3).contiguous()
        un = torch.stack(update_norm_all, dim=1).view(B, self.L, self.S, self.K, self.B).permute(0, 1, 4, 2, 3).contiguous()
        slots = torch.cat([base[:, None, :]] + slot_states, dim=1)
        names = ["input/base"] + slot_names
        return logits, Aux(
            family_gates=fam,
            write_gates=wr,
            update_norms=un,
            slot_states=slots,
            slot_names=names,
            input_skip_gates=torch.sigmoid(self.input_skip_logit.detach()).cpu(),
            mean_skip_gates=torch.sigmoid(self.mean_skip_logit.detach()).cpu(),
        )


# -----------------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------------


def retain_aux_grads(aux: Aux) -> None:
    for t in (aux.family_gates, aux.write_gates):
        if t is not None and t.requires_grad:
            t.retain_grad()


def grad_x_gate(t: torch.Tensor, scale: float) -> Optional[torch.Tensor]:
    if t is None or t.grad is None:
        return None
    return (torch.nan_to_num((t.grad.detach().float() / max(1.0, scale)).abs()) * torch.nan_to_num(t.detach().float().abs())).mean(dim=0)


def build_grad_report(model: CleanSequentialMatrixProgram, aux: Aux, scale: float) -> Dict:
    by: Dict[str, float] = {}
    top: List[Dict] = []
    fg = grad_x_gate(aux.family_gates, scale)
    if fg is not None:
        flat = fg.flatten()
        vals, idxs = torch.topk(flat, k=min(80, flat.numel()))
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            x = int(idx)
            f = x % model.F; x //= model.F
            k = x % model.K; x //= model.K
            s = x % model.S; x //= model.S
            b = x % model.B; x //= model.B
            l = x
            addr = f"L{l}.B{b}.S{s}.K{k}.family.{model.FAMILY_NAMES[f]}"
            by[addr] = float(v)
            top.append({"type": "family", "address": addr, "grad_x_gate": float(v)})
    wg = grad_x_gate(aux.write_gates, scale)
    if wg is not None:
        flat = wg.flatten()
        vals, idxs = torch.topk(flat, k=min(40, flat.numel()))
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            x = int(idx)
            k = x % model.K; x //= model.K
            s = x % model.S; x //= model.S
            b = x % model.B; x //= model.B
            l = x
            addr = f"L{l}.B{b}.S{s}.K{k}.write_gate"
            by[addr] = float(v)
            top.append({"type": "write_gate", "address": addr, "grad_x_gate": float(v)})
    top.sort(key=lambda r: r["grad_x_gate"], reverse=True)
    return {"definition": "grad_x_gate=mean(abs(dloss/dgate)*gate), divided by GradScaler scale", "top": top[:100], "by_address": by}


def make_reports(model: CleanSequentialMatrixProgram, aux: Aux, grad_report: Optional[Dict] = None) -> Tuple[Dict, List[Dict]]:
    rows: List[Dict] = []
    levels = {"family": [], "utility": [], "layer": []}
    fg = aux.family_gates.detach().float().mean(0)      # [L,B,S,K,F]
    wg = aux.write_gates.detach().float().mean(0)       # [L,B,S,K]
    un = aux.update_norms.detach().float().mean(0)      # [L,B,S,K]

    by_addr = (grad_report or {}).get("by_address", {})
    for l in range(model.L):
        levels["layer"].append({
            "address": f"L{l}",
            "input_skip_gate": float(aux.input_skip_gates[l]),
            "mean_skip_gate": float(aux.mean_skip_gates[l]),
        })
        rows.append({"type": "layer", "address": f"L{l}.input_skip_gate", "value": float(aux.input_skip_gates[l])})
        rows.append({"type": "layer", "address": f"L{l}.mean_skip_gate", "value": float(aux.mean_skip_gates[l])})
        for b in range(model.B):
            for s in range(model.S):
                step_family = fg[l, b, s]  # [K,F]
                step_write = wg[l, b, s]
                step_update = un[l, b, s]
                util = {
                    "address": f"L{l}.B{b}.S{s}",
                    "mean_write_gate": float(step_write.mean()),
                    "mean_update_norm": float(step_update.mean()),
                    "family_entropy": float(entropy(step_family, -1).mean()),
                    "top_family_mass": float(step_family.max(dim=-1).values.mean()),
                }
                levels["utility"].append(util)
                rows.append({"type": "utility", **util})
                for k in range(model.K):
                    levels["family"].append({"address": f"L{l}.B{b}.S{s}.K{k}", "top": top_named(model.FAMILY_NAMES, fg[l, b, s, k], 5)})
                    vals, idxs = torch.topk(fg[l, b, s, k], k=2)
                    margin = float(vals[0] - vals[1]) if vals.numel() > 1 else float(vals[0])
                    for fi, name in enumerate(model.FAMILY_NAMES):
                        addr = f"L{l}.B{b}.S{s}.K{k}.family.{name}"
                        row = {
                            "type": "family",
                            "address": addr,
                            "prob": float(fg[l, b, s, k, fi]),
                            "top_margin": margin,
                            "grad_x_gate": float(by_addr.get(addr, 0.0)),
                        }
                        rows.append(row)
                    waddr = f"L{l}.B{b}.S{s}.K{k}.write_gate"
                    rows.append({"type": "write_gate", "address": waddr, "value": float(wg[l, b, s, k]), "grad_x_gate": float(by_addr.get(waddr, 0.0))})
    summary = {
        "family_entropy_mean": float(entropy(fg, -1).mean()),
        "write_gate_mean": float(wg.mean()),
        "update_norm_mean": float(un.mean()),
        "top_family_mass_mean": float(fg.max(dim=-1).values.mean()),
        "input_skip_mean": float(aux.input_skip_gates.float().mean()),
        "mean_skip_mean": float(aux.mean_skip_gates.float().mean()),
    }
    return {"levels": levels, "summary": summary, "grad_report": grad_report}, rows


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------


def aux_losses(aux: Aux, args):
    fg = aux.family_gates.float()
    wg = aux.write_gates.float()
    fe = entropy(fg, -1).mean()
    write_mean = wg.mean()
    ent_floor = torch.tensor(float(args.family_entropy_floor), device=fg.device)
    write_target = torch.tensor(float(args.min_write_target), device=fg.device)
    return {
        "family_entropy": fe,
        "write_gate_mean": write_mean,
        "update_norm_mean": aux.update_norms.float().mean(),
        "family_entropy_floor_loss": F.relu(ent_floor - fe).pow(2),
        "write_activity_loss": F.relu(write_target - write_mean).pow(2),
    }


def train_epoch(model, loader, opt, scaler, device, dtype, epoch, args):
    model.train()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    progress = (epoch - 1) / max(1, args.epochs - 1)
    temp = args.family_temp_start + (args.family_temp_end - args.family_temp_start) * progress
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
            logits, aux = model(wav, family_temp=temp, return_aux=True)
            ce = F.cross_entropy(logits.float(), y)
            losses = aux_losses(aux, args)
            loss = ce
            loss = loss + float(args.lambda_family_entropy_floor) * losses["family_entropy_floor_loss"]
            loss = loss + float(args.lambda_write_activity) * losses["write_activity_loss"]
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
            family_gates=aux.family_gates.detach().cpu(),
            write_gates=aux.write_gates.detach().cpu(),
            update_norms=aux.update_norms.detach().cpu(),
            slot_states=aux.slot_states.detach().cpu(),
            slot_names=aux.slot_names,
            input_skip_gates=aux.input_skip_gates,
            mean_skip_gates=aux.mean_skip_gates,
        )
        if args.log_every and step % args.log_every == 0:
            print(
                f"epoch {epoch:03d} step {step:05d} loss={totals['loss']/max(1,totals['n']):.4f} "
                f"ce={totals['ce']/max(1,totals['n']):.4f} acc={100*totals['correct']/max(1,totals['n']):.2f}% "
                f"fam_ent={sums['family_entropy']/max(1,totals['n']):.3f} write={sums['write_gate_mean']/max(1,totals['n']):.3f}",
                flush=True,
            )
    out = {
        "loss": totals["loss"] / max(1, totals["n"]),
        "ce": totals["ce"] / max(1, totals["n"]),
        "acc": totals["correct"] / max(1, totals["n"]),
        "n": totals["n"],
        "family_temp": temp,
        "last_aux": last_aux,
        "grad_report": grad_rep,
    }
    for k, v in sums.items():
        out[k] = v / max(1, totals["n"])
    return out


@torch.no_grad()
def evaluate(model, loader, device, dtype, args):
    model.eval()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    total_loss = 0.0
    correct = 0
    n = 0
    last_aux = None
    conf = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long)
    for step, (wav, y) in enumerate(loader, 1):
        if args.max_val_batches and step > args.max_val_batches:
            break
        wav, y = wav.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, aux = model(wav, family_temp=args.family_temp_end, return_aux=True)
            loss = F.cross_entropy(logits.float(), y)
        pred = logits.argmax(-1)
        bs = y.numel()
        total_loss += float(loss.cpu()) * bs
        correct += int((pred == y).sum().cpu())
        n += bs
        conf += torch.bincount((y.cpu() * args.num_classes + pred.cpu()), minlength=args.num_classes ** 2).view(args.num_classes, args.num_classes)
        last_aux = Aux(
            family_gates=aux.family_gates.detach().cpu(),
            write_gates=aux.write_gates.detach().cpu(),
            update_norms=aux.update_norms.detach().cpu(),
            slot_states=aux.slot_states.detach().cpu(),
            slot_names=aux.slot_names,
            input_skip_gates=aux.input_skip_gates,
            mean_skip_gates=aux.mean_skip_gates,
        )
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
    model = CleanSequentialMatrixProgram(
        num_classes=len(classes), dim=args.dim, evidence_cells=args.evidence_cells,
        layers=args.layers, blocks=args.blocks, steps=args.steps, substeps=args.substeps,
        sample_rate=args.sample_rate, n_mels=args.n_mels, hop_length=args.hop_length,
        rank=args.rank, groups=args.groups, dropout=args.dropout,
    ).to(device)
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"class counts train={train_counts} val={val_counts}", flush=True)
    print(f"CleanSeqV3 params={sum(p.numel() for p in model.parameters())} L={args.layers} B={args.blocks} S={args.steps} K={args.substeps} F={len(model.FAMILY_NAMES)} device={device} amp={args.amp}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)
    fields = ["epoch", "train_loss", "train_ce", "train_acc", "val_loss", "val_acc", "best_acc", "family_entropy", "write_gate_mean", "update_norm_mean", "family_temp"]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    best = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, opt, scaler, device, dtype, epoch, args)
        va = evaluate(model, val_loader, device, dtype, args)
        if va["acc"] > best:
            best = va["acc"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best}, out_dir / "last.pt")
        report, rows = make_reports(model, va["last_aux"], tr.get("grad_report"))
        analysis = {
            "epoch": epoch,
            "train": {k: v for k, v in tr.items() if k not in ("last_aux", "grad_report")},
            "val": {"loss": va["loss"], "acc": va["acc"], "n": va["n"], "confusion": va["confusion"]},
            "best_acc": best,
            "best_epoch": best_epoch,
            "classes": classes,
            "train_counts": train_counts,
            "val_counts": val_counts,
            "program_report": report,
        }
        write_json(out_dir / f"analysis_epoch_{epoch:03d}.json", analysis)
        write_jsonl(out_dir / f"events_epoch_{epoch:03d}.jsonl", [{"epoch": epoch, **r} for r in rows])
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_ce": tr["ce"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "best_acc": best,
            "family_entropy": tr.get("family_entropy", 0.0),
            "write_gate_mean": tr.get("write_gate_mean", 0.0),
            "update_norm_mean": tr.get("update_norm_mean", 0.0),
            "family_temp": tr.get("family_temp", 0.0),
        }
        with (out_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch} fam_ent={tr.get('family_entropy',0):.3f} write={tr.get('write_gate_mean',0):.3f}", flush=True)
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
    p.add_argument("--family-temp-start", type=float, default=1.40)
    p.add_argument("--family-temp-end", type=float, default=0.90)
    p.add_argument("--lambda-family-entropy-floor", type=float, default=0.0)
    p.add_argument("--family-entropy-floor", type=float, default=0.40)
    p.add_argument("--lambda-write-activity", type=float, default=0.0)
    p.add_argument("--min-write-target", type=float, default=0.02)
    p.add_argument("--grad-analytics-every", type=int, default=50)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", default="./runs/step_program_v3_clean_seq")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
