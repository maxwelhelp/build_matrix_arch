#!/usr/bin/env python3
"""
StepProgram v2: clean differentiable matrix-program core with projected-topk.

Main purpose:
  - keep the clean StepProgram idea;
  - avoid computing all heavy primitives forever;
  - log full address-based analytics for automatic parsing.

Address format:
  L{layer}.B{block}.S{step}.P{slot}.{primitive}
  L{layer}.B{block}.S{step}.P{k}->P{k+1}.{transition}
  L{layer}.B{block}.S{step}.read.{source}
  L{layer}.B{block}.route.{source}
  class.{name}.read.{slot}

Execution modes:
  all             compute every primitive candidate, best for warmup/baseline
  projected_topk  cheap scores all primitives; fully compute selected heavy ops only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
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
# Utilities
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


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    q = p.float().clamp_min(1e-8)
    return -(q * q.log()).sum(dim=dim)


def top_named(names: Sequence[str], weights: torch.Tensor, k: int = 5) -> List[Dict]:
    w = torch.nan_to_num(weights.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if w.numel() == 0:
        return []
    vals, idxs = torch.topk(w, k=min(int(k), int(w.numel())))
    return [{"name": names[int(i)] if int(i) < len(names) else f"idx_{int(i)}", "value": float(v)} for v, i in zip(vals.tolist(), idxs.tolist())]


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


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
        phase = 0.13 * (idx % 17)
        wav = torch.sin(2.0 * math.pi * freq * self.t + phase) + 0.05 * torch.randn_like(self.t)
        return wav.unsqueeze(0), y


class SpeechCommandsFiltered(Dataset):
    def __init__(self, root: str, subset: str, classes: Sequence[str], limit: int = 0, download: bool = False):
        if torchaudio is None:
            raise RuntimeError("torchaudio is not available; use --synthetic for smoke tests")
        self.classes = list(classes)
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(root=root, subset=subset, download=download)
        self.keep: List[int] = []
        for i in range(len(self.ds)):
            try:
                label = Path(self.ds._walker[i]).parent.name
            except Exception:
                label = str(self.ds[i][2])
            if label in self.class_to_id:
                self.keep.append(i)
                if limit and len(self.keep) >= limit:
                    break

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
        train_ds = SyntheticSpeechLike(args.train_limit or 2048, len(classes), args.sample_rate, args.seconds)
        val_ds = SyntheticSpeechLike(args.val_limit or 512, len(classes), args.sample_rate, args.seconds)
    else:
        train_ds = SpeechCommandsFiltered(args.data_root, "training", classes, args.train_limit, args.download)
        val_ds = SpeechCommandsFiltered(args.data_root, "validation", classes, args.val_limit, args.download)
    collate = lambda b: collate_wavs(b, args.sample_rate, args.seconds)
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.pin_memory, collate_fn=collate)
    val = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.pin_memory, collate_fn=collate)
    return train, val, classes


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


@dataclass
class Aux:
    primitive_gates: torch.Tensor              # [B,L,N,S,K,O]
    primitive_scores: torch.Tensor             # [B,L,N,S,K,O]
    primitive_computed_mask: torch.Tensor       # [L,N,S,K,O]
    primitive_selected_mask: torch.Tensor       # [L,N,S,K,O]
    primitive_transition_gates: torch.Tensor   # [B,L,N,S,K-1,T]
    step_read_gates: torch.Tensor              # [B,L,N,S,R]
    layer_route_gates: torch.Tensor            # [B,L,N,LR]
    step_write_gates: torch.Tensor             # [B,L,N,S]
    global_write_gates: torch.Tensor           # [B,L,N,S]
    memory_write_gates: torch.Tensor           # [B,L,N,S]
    class_read: torch.Tensor                   # [B,C,slots]
    slot_states: torch.Tensor                  # [B,slots,D]
    logits: torch.Tensor
    slot_names: List[str]
    exec_stats: Dict[str, float]


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
        # Force frontend to fp32.  fp16 log-mel can underflow and create nonfinite loss.
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
            feats = torch.nan_to_num(feats[:, : self.evidence_cells], nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nan_to_num(self.norm(self.scalar_proj(feats.unsqueeze(-1)) + self.pos.view(1, self.evidence_cells, -1)), nan=0.0, posinf=0.0, neginf=0.0)


class StepProgramNet(nn.Module):
    PRIMITIVES = ["noop", "identity", "keep_state", "small_refine", "mlp", "matrix_mlp", "compare", "memory_read", "global_read", "normalize", "suppress"]
    COSTS = [0.05, 0.05, 0.05, 0.20, 1.00, 1.35, 0.75, 0.25, 0.25, 0.30, 0.45]
    SAFE = {0, 1, 2, 3, 9}
    HEAVY = {4, 5, 6, 10}
    TRANSITIONS = ["keep", "replace", "residual", "norm_residual", "product_gate", "compare_mix"]
    READS = ["state", "input", "prev_step", "all_prev_steps", "layer_route", "global", "memory", "task"]
    ROUTES = ["input_skip", "prev_same_block", "prev_layer_mean"]

    def __init__(self, num_classes: int, dim: int, evidence_cells: int, layers: int, blocks: int, steps: int, primitive_slots: int, global_cells: int, memory_cells: int, sample_rate: int, n_mels: int, hop_length: int, dropout: float):
        super().__init__()
        self.C, self.D, self.E = int(num_classes), int(dim), int(evidence_cells)
        self.L, self.N, self.S, self.K = int(layers), int(blocks), int(steps), int(primitive_slots)
        self.G, self.M = int(global_cells), int(memory_cells)
        self.O, self.T, self.R, self.LR = len(self.PRIMITIVES), len(self.TRANSITIONS), len(self.READS), len(self.ROUTES)
        self.evidence = AudioEvidence(sample_rate, n_mels, hop_length, evidence_cells, dim)
        self.base_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.in_q = nn.Linear(dim, dim, bias=False); self.in_k = nn.Linear(dim, dim, bias=False); self.in_v = nn.Linear(dim, dim, bias=False)
        self.global_k = nn.Linear(dim, dim, bias=False); self.global_v = nn.Linear(dim, dim, bias=False)
        self.memory_k = nn.Linear(dim, dim, bias=False); self.memory_v = nn.Linear(dim, dim, bias=False)
        self.layer_addr = nn.Parameter(torch.randn(self.L, dim) * 0.03)
        self.block_addr = nn.Parameter(torch.randn(self.N, dim) * 0.03)
        self.step_addr = nn.Parameter(torch.randn(self.S, dim) * 0.03)
        self.prim_addr = nn.Parameter(torch.randn(self.K, dim) * 0.03)
        self.global_init = nn.Parameter(torch.randn(self.G, dim) * 0.03)
        self.memory_init = nn.Parameter(torch.randn(self.M, dim) * 0.03)
        self.task_token = nn.Parameter(torch.randn(1, 1, dim) * 0.03)
        self.route_net = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, self.LR))
        self.read_net = nn.Sequential(nn.LayerNorm(dim * 5), nn.Linear(dim * 5, dim), nn.GELU(), nn.Linear(dim, self.R))
        self.prim_score = nn.Sequential(nn.LayerNorm(dim * 6), nn.Linear(dim * 6, dim), nn.GELU(), nn.Linear(dim, self.O))
        self.trans_net = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, self.T))
        self.write_net = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 1))
        self.mlp = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.small_refine = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.compare = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm_prim = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.suppress_gate = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.suppress_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.mat_q = nn.Linear(dim, dim, bias=False); self.mat_k = nn.Linear(dim, dim, bias=False); self.mat_v = nn.Linear(dim, dim, bias=False)
        self.mat_out = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.trans_compare = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.trans_gate = nn.Linear(dim * 3, dim)
        self.global_write_gate = nn.Linear(dim * 3, 1); self.memory_write_gate = nn.Linear(dim * 3, 1)
        self.global_write_val = nn.Linear(dim * 3, dim); self.memory_write_val = nn.Linear(dim * 3, dim)
        self.class_query = nn.Parameter(torch.randn(num_classes, dim) * 0.05)
        self.class_k = nn.Linear(dim, dim, bias=False); self.class_v = nn.Linear(dim, dim, bias=False)
        self.class_write = nn.Parameter(torch.randn(num_classes, dim) * 0.03)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.state_norm = nn.LayerNorm(dim); self.step_norm = nn.LayerNorm(dim); self.global_norm = nn.LayerNorm(dim); self.memory_norm = nn.LayerNorm(dim)
        self.register_buffer("cost", torch.tensor(self.COSTS, dtype=torch.float32))

    def _cell_attend(self, q: torch.Tensor, cells: torch.Tensor, kproj: nn.Module, vproj: nn.Module) -> torch.Tensor:
        score = torch.einsum("bnd,bmd->bnm", q, kproj(cells)) / math.sqrt(q.shape[-1])
        a = torch.softmax(score.float(), dim=-1).to(q.dtype)
        return torch.einsum("bnm,bmd->bnd", a, vproj(cells))

    def _input_ctx(self, state: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        score = torch.einsum("bnd,bed->bne", self.in_q(state), self.in_k(evidence)) / math.sqrt(state.shape[-1])
        a = torch.softmax(score.float(), dim=-1).to(state.dtype)
        return torch.einsum("bne,bed->bnd", a, self.in_v(evidence))

    def _op_output(self, op: int, x: torch.Tensor, state: torch.Tensor, read: torch.Tensor, blocks: torch.Tensor, glob: torch.Tensor, mem: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        if op == 0: return torch.zeros_like(x)
        if op == 1: return x
        if op == 2: return state
        if op == 3: return 0.10 * self.small_refine(torch.cat([x, read], dim=-1))
        if op == 4: return self.mlp(torch.cat([x, read, glob, mem], dim=-1))
        if op == 5:
            score = torch.einsum("bnd,bmd->bnm", self.mat_q(x), self.mat_k(blocks)) / math.sqrt(D)
            a = torch.softmax(score.float(), dim=-1).to(x.dtype)
            ctx = torch.einsum("bnm,bmd->bnd", a, self.mat_v(blocks))
            return self.mat_out(torch.cat([x, ctx, glob, mem], dim=-1))
        if op == 6: return self.compare(torch.cat([x - state, x * state, read, task], dim=-1))
        if op == 7: return mem
        if op == 8: return glob
        if op == 9: return self.norm_prim(torch.cat([F.layer_norm(x, (D,)), F.layer_norm(read, (D,)), F.layer_norm(glob + mem, (D,))], dim=-1))
        if op == 10:
            g = torch.sigmoid(self.suppress_gate(torch.cat([x, glob + mem, task], dim=-1)))
            return self.suppress_out(x - g * (glob + mem))
        raise ValueError(op)

    def _primitive_mix(self, x, state, read, blocks, glob, mem, task, logits, mode: str, epoch: int, warmup: int, topk: int, shadow_prob: float):
        B, N, D = x.shape
        probs_full = torch.softmax(logits.float(), dim=-1).to(x.dtype)
        selected = torch.zeros(N, self.O, device=x.device, dtype=torch.bool)
        computed = torch.zeros(N, self.O, device=x.device, dtype=torch.bool)
        if mode == "all" or int(epoch) <= int(warmup):
            selected[:] = True; computed[:] = True
            weights = probs_full
        else:
            # Shared selection per address/block across batch.  This avoids per-sample scatter chaos.
            score_mean = logits.detach().float().mean(dim=0)  # [N,O]
            for op in self.SAFE:
                selected[:, op] = True; computed[:, op] = True
            k = max(1, min(int(topk), self.O))
            vals, idxs = torch.topk(score_mean, k=k, dim=-1)
            selected.scatter_(1, idxs, True); computed.scatter_(1, idxs, True)
            if self.training and shadow_prob > 0 and random.random() < float(shadow_prob):
                # Add one extra non-selected heavy branch per block as a shadow candidate.
                heavy = torch.tensor(sorted(self.HEAVY), device=x.device, dtype=torch.long)
                for n in range(N):
                    candidates = [int(o) for o in heavy.tolist() if not bool(selected[n, int(o)])]
                    if candidates:
                        op = random.choice(candidates)
                        selected[n, op] = True; computed[n, op] = True
            mask = selected.view(1, N, self.O).to(x.dtype)
            weights = probs_full * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        out = torch.zeros_like(x)
        computed_count = 0
        full_count = self.O * N
        for op in range(self.O):
            if not bool(computed[:, op].any()):
                continue
            y = self._op_output(op, x, state, read, blocks, glob, mem, task)
            w = weights[:, :, op].unsqueeze(-1)
            if mode != "all" and int(epoch) > int(warmup):
                block_mask = computed[:, op].view(1, N, 1).to(x.dtype)
                y = y * block_mask
                computed_count += int(computed[:, op].sum().item())
            else:
                computed_count += N
            out = out + w * y
        stats = {"computed_ops": float(computed_count), "all_ops": float(full_count), "estimated_speedup": float(full_count / max(1, computed_count))}
        return out, weights, logits, selected, computed, stats

    def forward(self, wav: torch.Tensor, epoch: int = 1, primitive_temp: float = 1.0, read_temp: float = 1.0, primitive_exec_mode: str = "all", topk_primitives: int = 3, topk_warmup_epochs: int = 3, shadow_prob: float = 0.0, write_floor: float = 0.0, return_aux: bool = True):
        B, device, dtype, D = wav.shape[0], wav.device, wav.dtype, self.D
        evidence = self.evidence(wav)
        base = self.base_proj(evidence.mean(dim=1))
        block_addr = self.block_addr.to(device=device, dtype=dtype).view(1, self.N, D)
        prev_layer = self.state_norm(base[:, None, :] + block_addr)
        global_cells = self.global_init.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1) + 0.1 * base[:, None, :]
        memory_cells = self.memory_init.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1) + 0.1 * base[:, None, :]
        task = self.task_token.to(device=device, dtype=dtype).expand(B, self.N, D)

        pg_all=[]; ps_all=[]; pc_all=[]; psel_all=[]; tg_all=[]; rg_all=[]; lr_all=[]; wg_all=[]; gw_all=[]; mw_all=[]
        states=[]; slot_names=[]; exec_sum={"computed_ops":0.0,"all_ops":0.0}

        for l in range(self.L):
            state = self.state_norm(prev_layer + self.layer_addr[l].to(device=device, dtype=dtype).view(1,1,D) + block_addr)
            glob_mean = global_cells.mean(dim=1).unsqueeze(1).expand(-1,self.N,-1)
            mem_mean = memory_cells.mean(dim=1).unsqueeze(1).expand(-1,self.N,-1)
            prev_mean = prev_layer.mean(dim=1).unsqueeze(1).expand(-1,self.N,-1)
            route_logits = self.route_net(torch.cat([state, prev_layer, glob_mean, mem_mean], dim=-1))
            # Sequential basis prior.  The route is still differentiable, but
            # the default program is L(l-1).same_block -> L(l).same_block.
            # Order matches ROUTES: input_skip, prev_same_block, prev_layer_mean.
            route_prior = torch.tensor([-0.5, 2.0, -1.0], device=device, dtype=route_logits.dtype).view(1, 1, self.LR)
            route_gates = torch.softmax((route_logits + route_prior).float(), dim=-1).to(dtype)
            route_src = torch.stack([base[:,None,:].expand(-1,self.N,-1), prev_layer, prev_mean], dim=2)
            route_ctx = torch.einsum("bnr,bnrd->bnd", route_gates, route_src)
            lr_all.append(route_gates)
            prev_steps=[]
            for s in range(self.S):
                step_addr = self.step_addr[s].to(device=device,dtype=dtype).view(1,1,D)
                in_ctx = self._input_ctx(state + step_addr, evidence)
                glob_ctx = self._cell_attend(state + step_addr, global_cells, self.global_k, self.global_v)
                mem_ctx = self._cell_attend(state + step_addr, memory_cells, self.memory_k, self.memory_v)
                prev_step = prev_steps[-1] if prev_steps else torch.zeros_like(state)
                all_prev = torch.stack(prev_steps, dim=2).mean(dim=2) if prev_steps else torch.zeros_like(state)
                read_logits = self.read_net(torch.cat([state, in_ctx, route_ctx, glob_ctx, mem_ctx], dim=-1))
                # Phase priors for step specialization.
                # READS = state, input, prev_step, all_prev_steps, layer_route, global, memory, task
                if s == 0:
                    phase_prior = torch.tensor([1.0, 1.5, -1.0e4, -1.0e4, -0.2, -1.0, -1.0, 0.0], device=device, dtype=read_logits.dtype)
                elif s == 1:
                    phase_prior = torch.tensor([0.2, 0.3, 1.5, -0.2, 0.0, -0.5, -0.5, 0.0], device=device, dtype=read_logits.dtype)
                else:
                    phase_prior = torch.tensor([0.0, 0.0, 1.2, 0.6, -0.1, 0.2, 0.2, 0.1], device=device, dtype=read_logits.dtype)
                read_logits = read_logits + phase_prior.view(1, 1, self.R)
                if s == 0:
                    read_logits = read_logits.clone(); read_logits[...,2] = -1e4; read_logits[...,3] = -1e4
                read_gates = torch.softmax((read_logits / max(1e-4, read_temp)).float(), dim=-1).to(dtype)
                read_src = torch.stack([state, in_ctx, prev_step, all_prev, route_ctx, glob_ctx, mem_ctx, task], dim=2)
                read = torch.einsum("bnr,bnrd->bnd", read_gates, read_src)
                rg_all.append(read_gates)
                x = read
                pg_step=[]; ps_step=[]; psel_step=[]; pc_step=[]; tg_step=[]
                for kslot in range(self.K):
                    pa = self.prim_addr[kslot].to(device=device,dtype=dtype).view(1,1,D)
                    logits = self.prim_score(torch.cat([x + pa, state, read, task, glob_ctx, mem_ctx], dim=-1)) / max(1e-4, primitive_temp)
                    u, gates, scores, selected, computed, st = self._primitive_mix(x, state, read, state, glob_ctx, mem_ctx, task, logits, primitive_exec_mode, epoch, topk_warmup_epochs, topk_primitives, shadow_prob)
                    exec_sum["computed_ops"] += st["computed_ops"]; exec_sum["all_ops"] += st["all_ops"]
                    pg_step.append(gates); ps_step.append(scores); psel_step.append(selected); pc_step.append(computed)
                    if kslot + 1 < self.K:
                        trans_logits = self.trans_net(torch.cat([x, u, read, state], dim=-1))
                        trans_gates = torch.softmax(trans_logits.float(), dim=-1).to(dtype)
                        g = torch.sigmoid(self.trans_gate(torch.cat([x, u, read], dim=-1)))
                        trans_cands = torch.stack([x, u, x + g*u, self.step_norm(x + u), x * torch.sigmoid(u), self.trans_compare(torch.cat([x-u, x*u, read, state], dim=-1))], dim=2)
                        x = torch.einsum("bnt,bntd->bnd", trans_gates, trans_cands)
                        tg_step.append(trans_gates)
                    else:
                        x = u
                pg_step_t = torch.stack(pg_step, dim=2)
                safe_mass = pg_step_t[..., :3].mean(dim=2).sum(dim=-1).clamp(0,1)
                write_gate = torch.sigmoid(self.write_net(torch.cat([state, x, read, task], dim=-1))).squeeze(-1)
                write_gate = write_gate * (1.0 - 0.70 * safe_mass).clamp(0.05, 1.0)
                # Temporary anti-collapse write floor.  This is differentiable:
                # even when the learned write_gate is tiny, gradients still flow
                # through it because we use floor + (1-floor)*gate, not clamp.
                if write_floor > 0.0:
                    wf = float(max(0.0, min(0.25, write_floor)))
                    write_gate = wf + (1.0 - wf) * write_gate
                state = self.state_norm(state + write_gate.unsqueeze(-1) * x)

                gw_in = torch.cat([state, x, read], dim=-1); mw_in = torch.cat([state, x, mem_ctx], dim=-1)
                global_mass = pg_step_t[...,8].mean(dim=2).clamp(0,1)
                memory_mass = pg_step_t[...,7].mean(dim=2).clamp(0,1)
                gw_gate = 0.10 * torch.sigmoid(self.global_write_gate(gw_in)).squeeze(-1) * (0.05 + 0.95 * global_mass)
                mw_gate = 0.10 * torch.sigmoid(self.memory_write_gate(mw_in)).squeeze(-1) * (0.05 + 0.95 * memory_mass)
                global_delta = torch.einsum("bng,bnd->bgd", torch.softmax(torch.einsum("bnd,bgd->bng", state, global_cells)/math.sqrt(D), dim=-1).to(dtype) * gw_gate.unsqueeze(-1), self.global_write_val(gw_in)) / max(1,self.N)
                memory_delta = torch.einsum("bnm,bnd->bmd", torch.softmax(torch.einsum("bnd,bmd->bnm", state, memory_cells)/math.sqrt(D), dim=-1).to(dtype) * mw_gate.unsqueeze(-1), self.memory_write_val(mw_in)) / max(1,self.N)
                global_cells = self.global_norm(global_cells + global_delta); memory_cells = self.memory_norm(memory_cells + memory_delta)

                pg_all.append(pg_step_t); ps_all.append(torch.stack(ps_step, dim=2)); psel_all.append(torch.stack(psel_step, dim=1)); pc_all.append(torch.stack(pc_step, dim=1))
                tg_all.append(torch.stack(tg_step, dim=2) if tg_step else torch.empty(B,self.N,0,self.T,device=device,dtype=dtype))
                wg_all.append(write_gate); gw_all.append(gw_gate); mw_all.append(mw_gate)
                states.append(state); prev_steps.append(state)
                for b in range(self.N): slot_names.append(f"L{l}.B{b}.S{s}")
            prev_layer = state

        slot_states = torch.cat([base[:,None,:]] + states, dim=1)
        slot_names_full = ["input/base"] + slot_names
        q = self.class_query.to(device=device,dtype=dtype).view(1,self.C,D).expand(B,-1,-1)
        cls_score = torch.einsum("bcd,btd->bct", q, self.class_k(slot_states)) / math.sqrt(D)
        cls_attn = torch.softmax(cls_score.float(), dim=-1).to(dtype)
        cls_ctx = torch.einsum("bct,btd->bcd", cls_attn, self.class_v(slot_states))
        logits = torch.einsum("bcd,cd->bc", cls_ctx, self.class_write.to(device=device,dtype=dtype)) + self.bias.to(device=device,dtype=dtype)
        if not return_aux: return logits, None

        def stack_blns(xs, extra_shape=()):
            return torch.stack(xs, dim=1).view(B,self.L,self.S,self.N,*extra_shape).permute(0,1,3,2,*range(4,4+len(extra_shape))).contiguous()
        primitive_gates = stack_blns(pg_all, (self.K,self.O))
        primitive_scores = stack_blns(ps_all, (self.K,self.O))
        primitive_transition_gates = stack_blns(tg_all, (max(0,self.K-1),self.T))
        step_read_gates = stack_blns(rg_all, (self.R,))
        step_write_gates = stack_blns(wg_all)
        global_write_gates = stack_blns(gw_all)
        memory_write_gates = stack_blns(mw_all)
        primitive_selected_mask = torch.stack(psel_all, dim=0).view(self.L,self.S,self.N,self.K,self.O).permute(0,2,1,3,4).contiguous()
        primitive_computed_mask = torch.stack(pc_all, dim=0).view(self.L,self.S,self.N,self.K,self.O).permute(0,2,1,3,4).contiguous()
        exec_sum["estimated_speedup"] = float(exec_sum["all_ops"] / max(1.0, exec_sum["computed_ops"]))
        return logits, Aux(primitive_gates, primitive_scores, primitive_computed_mask, primitive_selected_mask, primitive_transition_gates, step_read_gates, torch.stack(lr_all,dim=1), step_write_gates, global_write_gates, memory_write_gates, cls_attn, slot_states, logits, slot_names_full, exec_sum)


# -----------------------------------------------------------------------------
# Reports and train loop
# -----------------------------------------------------------------------------


def aux_losses(model: StepProgramNet, aux: Aux, epoch: int, epochs: int, args):
    prog = min(1.0, max(0.0, (epoch - 1) / max(1, epochs - 1)))

    # Delayed regularization: cost/write/entropy penalties were the cause of
    # the v2 identity/no-write collapse.  They should not dominate before the
    # program has nonzero writes and nontrivial primitive mass.
    reg_start = max(1, int(getattr(args, "reg_warmup_epochs", 5)))
    if epoch <= reg_start:
        reg_w = 0.0
    else:
        reg_w = min(1.0, (epoch - reg_start) / max(1, epochs - reg_start))

    # Anti-collapse support.  Keep some activity/exploration alive while the
    # program learns to use its steps.  After the early phase, keep a small tail
    # so the system does not instantly harden back to identity.
    anti_epochs = max(1, int(getattr(args, "anti_collapse_epochs", 6)))
    anti_tail = float(getattr(args, "anti_collapse_tail", 0.25))
    anti_w = 1.0 if epoch <= anti_epochs else anti_tail

    pg = aux.primitive_gates.float()
    prim_entropy = entropy(pg, -1).mean()
    trans_entropy = entropy(aux.primitive_transition_gates.float(), -1).mean() if aux.primitive_transition_gates.numel() else torch.zeros((), device=pg.device)
    read_entropy = entropy(aux.step_read_gates.float(), -1).mean()
    write_mean = aux.step_write_gates.float().mean()
    cost = (pg * model.cost.to(pg.device).view(1,1,1,1,1,-1)).sum(dim=-1).mean()

    # Nontrivial means: not noop, not identity, not keep_state.
    nontrivial_mass = pg[..., 3:].sum(dim=-1).mean()
    safe_mass = pg[..., :3].sum(dim=-1).mean()

    min_write_target = torch.tensor(float(getattr(args, "min_write_target", 0.02)), device=pg.device)
    min_nontrivial = torch.tensor(float(getattr(args, "min_nontrivial_mass", 0.10)), device=pg.device)
    prim_ent_floor = torch.tensor(float(getattr(args, "primitive_entropy_floor", 0.35)), device=pg.device)
    read_ent_floor = torch.tensor(float(getattr(args, "read_entropy_floor", 0.35)), device=pg.device)
    trans_ent_floor = torch.tensor(float(getattr(args, "transition_entropy_floor", 0.25)), device=pg.device)

    write_activity_loss = F.relu(min_write_target - write_mean).pow(2)
    nontrivial_activity_loss = F.relu(min_nontrivial - nontrivial_mass).pow(2)
    primitive_entropy_floor_loss = F.relu(prim_ent_floor - prim_entropy).pow(2)
    read_entropy_floor_loss = F.relu(read_ent_floor - read_entropy).pow(2)
    transition_entropy_floor_loss = F.relu(trans_ent_floor - trans_entropy).pow(2)

    return {
        "cost_loss": cost,
        "primitive_entropy": prim_entropy,
        "transition_entropy": trans_entropy,
        "read_entropy": read_entropy,
        "write_l1": write_mean,
        "nontrivial_mass": nontrivial_mass,
        "safe_mass": safe_mass,
        "write_activity_loss": write_activity_loss,
        "nontrivial_activity_loss": nontrivial_activity_loss,
        "primitive_entropy_floor_loss": primitive_entropy_floor_loss,
        "read_entropy_floor_loss": read_entropy_floor_loss,
        "transition_entropy_floor_loss": transition_entropy_floor_loss,
        "lambda_cost_eff": torch.tensor(float(args.lambda_cost) * reg_w, device=pg.device),
        "lambda_primitive_entropy_eff": torch.tensor(float(args.lambda_primitive_entropy) * reg_w, device=pg.device),
        "lambda_transition_entropy_eff": torch.tensor(float(args.lambda_transition_entropy) * reg_w, device=pg.device),
        "lambda_read_entropy_eff": torch.tensor(float(args.lambda_read_entropy) * reg_w, device=pg.device),
        "lambda_write_l1_eff": torch.tensor(float(args.lambda_write_l1) * reg_w, device=pg.device),
        "lambda_write_activity_eff": torch.tensor(float(getattr(args, "lambda_write_activity", 0.05)) * anti_w, device=pg.device),
        "lambda_nontrivial_activity_eff": torch.tensor(float(getattr(args, "lambda_nontrivial_activity", 0.05)) * anti_w, device=pg.device),
        "lambda_primitive_entropy_floor_eff": torch.tensor(float(getattr(args, "lambda_primitive_entropy_floor", 0.02)) * anti_w, device=pg.device),
        "lambda_read_entropy_floor_eff": torch.tensor(float(getattr(args, "lambda_read_entropy_floor", 0.01)) * anti_w, device=pg.device),
        "lambda_transition_entropy_floor_eff": torch.tensor(float(getattr(args, "lambda_transition_entropy_floor", 0.005)) * anti_w, device=pg.device),
        "reg_w": torch.tensor(float(reg_w), device=pg.device),
        "anti_w": torch.tensor(float(anti_w), device=pg.device),
        "sharpen": torch.tensor(prog, device=pg.device),
    }

def retain_aux_grads(aux: Aux):
    for t in (aux.primitive_gates, aux.primitive_transition_gates, aux.step_read_gates, aux.layer_route_gates, aux.step_write_gates, aux.class_read):
        if t is not None and t.requires_grad: t.retain_grad()


def grad_x_gate(t: torch.Tensor, scale: float) -> Optional[torch.Tensor]:
    if t is None or t.grad is None: return None
    return (torch.nan_to_num((t.grad.detach().float()/max(1.0,scale)).abs()) * torch.nan_to_num(t.detach().float().abs())).mean(dim=0)


def make_reports(model: StepProgramNet, aux: Aux, classes: Sequence[str], grad_report: Optional[Dict] = None):
    rows=[]; levels={"primitive":[],"transition":[],"step_read":[],"layer_route":[],"utility":[],"class_read":[],"suspicious":[]}
    pg=aux.primitive_gates.detach().float().mean(0); ps=aux.primitive_scores.detach().float().mean(0)
    tg=aux.primitive_transition_gates.detach().float().mean(0); rg=aux.step_read_gates.detach().float().mean(0); lr=aux.layer_route_gates.detach().float().mean(0)
    wg=aux.step_write_gates.detach().float().mean(0); gw=aux.global_write_gates.detach().float().mean(0); mw=aux.memory_write_gates.detach().float().mean(0)
    cls=aux.class_read.detach().float().mean(0); slot_use=cls.mean(0)
    route_name_to_idx = {name: i for i, name in enumerate(model.ROUTES)}
    seq_metrics = {
        "sequential_route_mass": float(lr[..., route_name_to_idx.get("prev_same_block", 0)].mean()) if "prev_same_block" in route_name_to_idx else 0.0,
        "input_skip_mass": float(lr[..., route_name_to_idx.get("input_skip", 0)].mean()) if "input_skip" in route_name_to_idx else 0.0,
        "prev_layer_mean_mass": float(lr[..., route_name_to_idx.get("prev_layer_mean", 0)].mean()) if "prev_layer_mean" in route_name_to_idx else 0.0,
        "step_input_mass": float((rg[..., 0] + rg[..., 1]).mean()),
        "step_prev_mass": float((rg[..., 2] + rg[..., 3]).mean()),
        "step_layer_route_mass": float(rg[..., 4].mean()),
        "step_register_mass": float((rg[..., 5] + rg[..., 6]).mean()),
        "identity_mass": float(pg[..., 1].mean()),
        "safe_mass": float(pg[..., :3].sum(dim=-1).mean()),
        "nontrivial_mass": float(pg[..., 3:].sum(dim=-1).mean()),
        "mean_write_gate": float(wg.mean()),
    }
    for l in range(model.L):
      for b in range(model.N):
        route_top=top_named(model.ROUTES, lr[l,b], 5); levels["layer_route"].append({"address":f"L{l}.B{b}.route","top":route_top})
        for ri,name in enumerate(model.ROUTES): rows.append({"type":"layer_route","address":f"L{l}.B{b}.route.{name}","prob":float(lr[l,b,ri])})
        for s in range(model.S):
          step_addr=f"L{l}.B{b}.S{s}"; slot_idx=1+((l*model.S+s)*model.N+b); class_mass=float(slot_use[slot_idx]) if slot_idx < slot_use.numel() else 0.0
          safe=float(pg[l,b,s,:,:3].mean().clamp(0,1)); nontriv=float(pg[l,b,s,:,4:].sum(-1).mean().clamp(0,1)); expensive=float(pg[l,b,s,:,[4,5,6,10]].sum(-1).mean().clamp(0,1))
          util=float(wg[l,b,s]*class_mass)
          u={"address":step_addr,"write_gate":float(wg[l,b,s]),"class_read_mass":class_mass,"safe_mass":safe,"nontrivial_mass":nontriv,"expensive_mass":expensive,"global_write_gate":float(gw[l,b,s]),"memory_write_gate":float(mw[l,b,s]),"utility_proxy":util}
          levels["utility"].append(u); rows.append({"type":"utility",**u})
          if class_mass>0.025 and safe>0.55: levels["suspicious"].append({"address":step_addr,"kind":"class_reads_safe_slot",**u})
          if expensive>0.50 and util<0.002: levels["suspicious"].append({"address":step_addr,"kind":"expensive_low_utility",**u})
          levels["step_read"].append({"address":f"{step_addr}.read","top":top_named(model.READS, rg[l,b,s], 5)})
          for r,name in enumerate(model.READS): rows.append({"type":"step_read","address":f"{step_addr}.read.{name}","prob":float(rg[l,b,s,r])})
          for k in range(model.K):
            top=top_named(model.PRIMITIVES, pg[l,b,s,k], 5); levels["primitive"].append({"address":f"{step_addr}.P{k}","top":top})
            probs=pg[l,b,s,k]; vals,idx=torch.topk(probs,k=2); margin=float(vals[0]-vals[1]) if len(vals)>1 else float(vals[0])
            for o,name in enumerate(model.PRIMITIVES):
              rows.append({"type":"primitive","address":f"{step_addr}.P{k}.{name}","prob":float(probs[o]),"score":float(ps[l,b,s,k,o]),"selected":bool(aux.primitive_selected_mask[l,b,s,k,o]),"computed":bool(aux.primitive_computed_mask[l,b,s,k,o]),"cost":float(model.COSTS[o]),"top_margin":margin})
          for k in range(max(0,model.K-1)):
            levels["transition"].append({"address":f"{step_addr}.P{k}->P{k+1}","top":top_named(model.TRANSITIONS,tg[l,b,s,k],4)})
            for ti,name in enumerate(model.TRANSITIONS): rows.append({"type":"transition","address":f"{step_addr}.P{k}->P{k+1}.{name}","prob":float(tg[l,b,s,k,ti])})
    for ci,cname in enumerate(classes):
      levels["class_read"].append({"class":cname,"top":top_named(aux.slot_names, cls[ci], 8)})
      for si,sname in enumerate(aux.slot_names): rows.append({"type":"class_read","address":f"class.{cname}.read.{sname}","prob":float(cls[ci,si])})
    if grad_report:
      by_addr=grad_report.get("by_address",{})
      for r in rows:
        if r["address"] in by_addr: r["grad_x_gate"] = by_addr[r["address"]]
    return {"levels":levels,"exec_stats":aux.exec_stats,"sequential_metrics":seq_metrics,"grad_report":grad_report}, rows


def build_grad_report(model: StepProgramNet, aux: Aux, classes: Sequence[str], scale: float):
    by={}; top=[]
    groups=[("primitive",grad_x_gate(aux.primitive_gates,scale),model.PRIMITIVES), ("transition",grad_x_gate(aux.primitive_transition_gates,scale),model.TRANSITIONS), ("step_read",grad_x_gate(aux.step_read_gates,scale),model.READS), ("layer_route",grad_x_gate(aux.layer_route_gates,scale),model.ROUTES)]
    for typ,arr,names in groups:
      if arr is None: continue
      flat=arr.flatten(); vals,idx=torch.topk(flat,k=min(40,flat.numel()))
      for v,ii in zip(vals.tolist(),idx.tolist()):
        coords=[]; x=int(ii)
        for size in reversed(arr.shape): coords.append(x%size); x//=size
        coords=list(reversed(coords))
        if typ=="primitive": addr=f"L{coords[0]}.B{coords[1]}.S{coords[2]}.P{coords[3]}.{names[coords[4]]}"
        elif typ=="transition": addr=f"L{coords[0]}.B{coords[1]}.S{coords[2]}.P{coords[3]}->P{coords[3]+1}.{names[coords[4]]}"
        elif typ=="step_read": addr=f"L{coords[0]}.B{coords[1]}.S{coords[2]}.read.{names[coords[3]]}"
        else: addr=f"L{coords[0]}.B{coords[1]}.route.{names[coords[2]]}"
        by[addr]=float(v); top.append({"type":typ,"address":addr,"grad_x_gate":float(v)})
    arr=grad_x_gate(aux.step_write_gates,scale)
    if arr is not None:
      flat=arr.flatten(); vals,idx=torch.topk(flat,k=min(20,flat.numel()))
      for v,ii in zip(vals.tolist(),idx.tolist()):
        l,b,s = int(ii)//(model.N*model.S), (int(ii)//model.S)%model.N, int(ii)%model.S
        addr=f"L{l}.B{b}.S{s}.write_gate"; by[addr]=float(v); top.append({"type":"write_gate","address":addr,"grad_x_gate":float(v)})
    arr=grad_x_gate(aux.class_read,scale)
    if arr is not None:
      flat=arr.flatten(); vals,idx=torch.topk(flat,k=min(30,flat.numel()))
      for v,ii in zip(vals.tolist(),idx.tolist()):
        c=int(ii)//len(aux.slot_names); si=int(ii)%len(aux.slot_names); cname=classes[c] if c<len(classes) else f"class_{c}"
        addr=f"class.{cname}.read.{aux.slot_names[si]}"; by[addr]=float(v); top.append({"type":"class_read","address":addr,"grad_x_gate":float(v)})
    top.sort(key=lambda x:x["grad_x_gate"], reverse=True)
    return {"definition":"grad_x_gate=mean(abs(dloss/dgate)*gate), divided by GradScaler scale", "top":top[:100], "by_address":by}


def train_epoch(model, loader, opt, scaler, device, dtype, epoch, args, classes):
    model.train(); use_amp=device.startswith("cuda") and dtype!=torch.float32
    totals={"loss":0.0,"ce":0.0,"correct":0,"n":0}; sums={}; last_aux=None; grad_rep=None; t0=time.time()
    progress=(epoch-1)/max(1,args.epochs-1); ptemp=args.primitive_temp_start+(args.primitive_temp_end-args.primitive_temp_start)*progress; rtemp=args.read_temp_start+(args.read_temp_end-args.read_temp_start)*progress
    wf_prog = min(1.0, max(0.0, (epoch - 1) / max(1, int(getattr(args, "write_floor_epochs", 6)) - 1)))
    write_floor = float(getattr(args, "write_floor_start", 0.02)) + (float(getattr(args, "write_floor_end", 0.0)) - float(getattr(args, "write_floor_start", 0.02))) * wf_prog
    for step,(wav,y) in enumerate(loader,1):
      if args.max_train_batches and step>args.max_train_batches: break
      wav,y=wav.to(device,non_blocking=True),y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
      with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
        logits,aux=model(wav,epoch=epoch,primitive_temp=ptemp,read_temp=rtemp,primitive_exec_mode=args.primitive_exec_mode,topk_primitives=args.topk_primitives,topk_warmup_epochs=args.topk_warmup_epochs,shadow_prob=args.shadow_prob,write_floor=write_floor,return_aux=True)
        ce=F.cross_entropy(logits.float(),y); losses=aux_losses(model,aux,epoch,args.epochs,args)
        loss=ce + losses["lambda_cost_eff"]*losses["cost_loss"] + losses["lambda_primitive_entropy_eff"]*losses["primitive_entropy"] + losses["lambda_transition_entropy_eff"]*losses["transition_entropy"] + losses["lambda_read_entropy_eff"]*losses["read_entropy"] + losses["lambda_write_l1_eff"]*losses["write_l1"]
        loss = loss + losses["lambda_write_activity_eff"] * losses["write_activity_loss"]
        loss = loss + losses["lambda_nontrivial_activity_eff"] * losses["nontrivial_activity_loss"]
        loss = loss + losses["lambda_primitive_entropy_floor_eff"] * losses["primitive_entropy_floor_loss"]
        loss = loss + losses["lambda_read_entropy_floor_eff"] * losses["read_entropy_floor_loss"]
        loss = loss + losses["lambda_transition_entropy_floor_eff"] * losses["transition_entropy_floor_loss"]
      do_grad=bool(args.grad_analytics_every and step%args.grad_analytics_every==0)
      if do_grad: retain_aux_grads(aux)
      if not torch.isfinite(loss): print("NONFINITE_LOSS skip", flush=True); continue
      scale=float(scaler.get_scale()) if hasattr(scaler,"get_scale") else 1.0
      scaler.scale(loss).backward()
      if do_grad: grad_rep=build_grad_report(model,aux,classes,scale)
      if args.grad_clip>0:
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      scaler.step(opt); scaler.update()
      bs=y.numel(); totals["loss"]+=float(loss.detach().cpu())*bs; totals["ce"]+=float(ce.detach().cpu())*bs; totals["correct"]+=int((logits.argmax(-1)==y).sum().detach().cpu()); totals["n"]+=bs
      for k,v in losses.items(): sums[k]=sums.get(k,0.0)+float(v.detach().cpu())*bs
      last_aux=Aux(*(getattr(aux,f).detach().cpu() if isinstance(getattr(aux,f),torch.Tensor) else getattr(aux,f) for f in aux.__dataclass_fields__))
      if args.log_every and step%args.log_every==0: print(f"epoch {epoch:03d} step {step:05d} loss={totals['loss']/max(1,totals['n']):.4f} ce={totals['ce']/max(1,totals['n']):.4f} acc={100*totals['correct']/max(1,totals['n']):.2f}% seen={totals['n']} t={time.time()-t0:.1f}s exec_speedup={aux.exec_stats.get('estimated_speedup',1):.2f}", flush=True)
    out={"loss":totals["loss"]/max(1,totals["n"]),"ce":totals["ce"]/max(1,totals["n"]),"acc":totals["correct"]/max(1,totals["n"]),"n":totals["n"],"last_aux":last_aux,"grad_report":grad_rep,"primitive_temp":ptemp,"read_temp":rtemp,"write_floor":write_floor}
    for k,v in sums.items(): out[k]=v/max(1,totals["n"])
    return out

@torch.no_grad()
def evaluate(model, loader, device, dtype, args, classes, epoch):
    model.eval(); use_amp=device.startswith("cuda") and dtype!=torch.float32
    total_loss=correct=n=0; conf=torch.zeros(len(classes),len(classes),dtype=torch.long); last_aux=None
    for step,(wav,y) in enumerate(loader,1):
      if args.max_val_batches and step>args.max_val_batches: break
      wav,y=wav.to(device,non_blocking=True),y.to(device,non_blocking=True)
      wf_prog = min(1.0, max(0.0, (epoch - 1) / max(1, int(getattr(args, "write_floor_epochs", 6)) - 1)))
      write_floor = float(getattr(args, "write_floor_start", 0.02)) + (float(getattr(args, "write_floor_end", 0.0)) - float(getattr(args, "write_floor_start", 0.02))) * wf_prog
      with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp): logits,aux=model(wav,epoch=epoch,primitive_temp=args.primitive_temp_end,read_temp=args.read_temp_end,primitive_exec_mode=args.primitive_exec_mode,topk_primitives=args.topk_primitives,topk_warmup_epochs=args.topk_warmup_epochs,shadow_prob=0.0,write_floor=write_floor,return_aux=True); loss=F.cross_entropy(logits.float(),y)
      pred=logits.argmax(-1); bs=y.numel(); total_loss+=float(loss.cpu())*bs; correct+=int((pred==y).sum().cpu()); n+=bs; conf+=torch.bincount((y.cpu()*len(classes)+pred.cpu()),minlength=len(classes)**2).view(len(classes),len(classes))
      last_aux=Aux(*(getattr(aux,f).detach().cpu() if isinstance(getattr(aux,f),torch.Tensor) else getattr(aux,f) for f in aux.__dataclass_fields__))
    return {"loss":total_loss/max(1,n),"acc":correct/max(1,n),"n":n,"confusion":conf,"last_aux":last_aux}


def run(args):
    set_seed(args.seed); device=args.device if args.device!="cuda" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"): torch.backends.cudnn.benchmark=True; torch.set_float32_matmul_precision("high")
    dtype=amp_dtype(args.amp); out=ensure_dir(Path(args.out_dir)); train_loader,val_loader,classes=make_loaders(args)
    model=StepProgramNet(len(classes),args.dim,args.evidence_cells,args.layers,args.blocks,args.steps,args.primitive_slots,args.global_cells,args.memory_cells,args.sample_rate,args.n_mels,args.hop_length,args.dropout).to(device)
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"StepProgramV2 params={sum(p.numel() for p in model.parameters())} L={args.layers} B={args.blocks} S={args.steps} K={args.primitive_slots} O={len(model.PRIMITIVES)} mode={args.primitive_exec_mode} device={device} amp={args.amp}", flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,betas=(0.9,0.95)); scaler=torch.amp.GradScaler("cuda",enabled=device.startswith("cuda") and dtype==torch.float16)
    fields=["epoch","train_loss","train_ce","train_acc","val_loss","val_acc","best_acc","cost_loss","primitive_entropy","transition_entropy","read_entropy","write_l1","nontrivial_mass","safe_mass","write_activity_loss","nontrivial_activity_loss","primitive_entropy_floor_loss","read_entropy_floor_loss","transition_entropy_floor_loss","lambda_cost_eff","lambda_write_l1_eff","lambda_write_activity_eff","lambda_nontrivial_activity_eff","lambda_primitive_entropy_floor_eff","lambda_read_entropy_floor_eff","lambda_transition_entropy_floor_eff","reg_w","anti_w","write_floor","exec_speedup"]
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=fields).writeheader()
    best=-1; best_epoch=0
    for epoch in range(1,args.epochs+1):
      tr=train_epoch(model,train_loader,opt,scaler,device,dtype,epoch,args,classes); va=evaluate(model,val_loader,device,dtype,args,classes,epoch)
      if va["acc"]>best: best=va["acc"]; best_epoch=epoch; torch.save({"model":model.state_dict(),"args":vars(args),"classes":classes,"epoch":epoch,"best_acc":best},out/"best.pt")
      torch.save({"model":model.state_dict(),"args":vars(args),"classes":classes,"epoch":epoch,"best_acc":best},out/"last.pt")
      rep,rows=make_reports(model,va["last_aux"],classes,tr.get("grad_report"))
      analysis={"epoch":epoch,"train":{k:v for k,v in tr.items() if k not in ("last_aux","grad_report")},"val":{"loss":va["loss"],"acc":va["acc"],"n":va["n"]},"best_acc":best,"best_epoch":best_epoch,"program_report":rep}
      write_json(out/f"analysis_epoch_{epoch:03d}.json",analysis); write_jsonl(out/f"events_epoch_{epoch:03d}.jsonl",[{"epoch":epoch,**r} for r in rows])
      row={"epoch":epoch,"train_loss":tr["loss"],"train_ce":tr["ce"],"train_acc":tr["acc"],"val_loss":va["loss"],"val_acc":va["acc"],"best_acc":best,"exec_speedup":rep["exec_stats"].get("estimated_speedup",1.0)}
      for f in fields: row.setdefault(f,tr.get(f,0.0))
      with (out/"metrics.csv").open("a",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=fields).writerow(row)
      print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch} exec_speedup={rep['exec_stats'].get('estimated_speedup',1):.2f}", flush=True)
    write_json(out/"final_report.json",{"best_acc":best,"best_epoch":best_epoch,"args":vars(args),"classes":classes})


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",default="./data/speechcommands"); p.add_argument("--download",action="store_true"); p.add_argument("--synthetic",action="store_true")
    p.add_argument("--classes",default="yes,no,up,down,left,right,on,off,stop,go"); p.add_argument("--train-limit",type=int,default=12000); p.add_argument("--val-limit",type=int,default=2000)
    p.add_argument("--seconds",type=float,default=1.0); p.add_argument("--sample-rate",type=int,default=16000); p.add_argument("--n-mels",type=int,default=64); p.add_argument("--hop-length",type=int,default=160)
    p.add_argument("--dim",type=int,default=96); p.add_argument("--evidence-cells",type=int,default=48); p.add_argument("--layers",type=int,default=3); p.add_argument("--blocks",type=int,default=4); p.add_argument("--steps",type=int,default=3); p.add_argument("--primitive-slots",type=int,default=2); p.add_argument("--global-cells",type=int,default=3); p.add_argument("--memory-cells",type=int,default=6); p.add_argument("--dropout",type=float,default=0.05)
    p.add_argument("--primitive-exec-mode",choices=["all","projected_topk"],default="projected_topk"); p.add_argument("--topk-primitives",type=int,default=3); p.add_argument("--topk-warmup-epochs",type=int,default=2); p.add_argument("--shadow-prob",type=float,default=0.10)
    p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--eval-batch-size",type=int,default=256); p.add_argument("--workers",type=int,default=4); p.add_argument("--pin-memory",action="store_true")
    p.add_argument("--lr",type=float,default=5e-4); p.add_argument("--weight-decay",type=float,default=0.01); p.add_argument("--grad-clip",type=float,default=0.7); p.add_argument("--amp",choices=["fp16","bf16","fp32","off"],default="fp16"); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--primitive-temp-start",type=float,default=1.50); p.add_argument("--primitive-temp-end",type=float,default=0.70); p.add_argument("--read-temp-start",type=float,default=1.25); p.add_argument("--read-temp-end",type=float,default=0.85)
    p.add_argument("--lambda-cost",type=float,default=0.005); p.add_argument("--lambda-primitive-entropy",type=float,default=0.002); p.add_argument("--lambda-transition-entropy",type=float,default=0.001); p.add_argument("--lambda-read-entropy",type=float,default=0.001); p.add_argument("--lambda-write-l1",type=float,default=0.002)
    p.add_argument("--reg-warmup-epochs",type=int,default=5)
    p.add_argument("--anti-collapse-epochs",type=int,default=6); p.add_argument("--anti-collapse-tail",type=float,default=0.25)
    p.add_argument("--lambda-write-activity",type=float,default=0.05); p.add_argument("--min-write-target",type=float,default=0.02)
    p.add_argument("--lambda-nontrivial-activity",type=float,default=0.05); p.add_argument("--min-nontrivial-mass",type=float,default=0.10)
    p.add_argument("--lambda-primitive-entropy-floor",type=float,default=0.02); p.add_argument("--primitive-entropy-floor",type=float,default=0.35)
    p.add_argument("--lambda-read-entropy-floor",type=float,default=0.01); p.add_argument("--read-entropy-floor",type=float,default=0.35)
    p.add_argument("--lambda-transition-entropy-floor",type=float,default=0.005); p.add_argument("--transition-entropy-floor",type=float,default=0.25)
    p.add_argument("--write-floor-start",type=float,default=0.02); p.add_argument("--write-floor-end",type=float,default=0.0); p.add_argument("--write-floor-epochs",type=int,default=6)
    p.add_argument("--grad-analytics-every",type=int,default=50); p.add_argument("--max-train-batches",type=int,default=0); p.add_argument("--max-val-batches",type=int,default=0); p.add_argument("--log-every",type=int,default=50); p.add_argument("--out-dir",default="./runs/step_program_v2_projected_topk")
    return p

if __name__ == "__main__":
    run(parser().parse_args())
