#!/usr/bin/env python3
"""
V4 simple matrix sanity baseline.

Purpose: prove that SpeechCommands data + log-mel frontend + final head can learn
before adding architecture search, routers, growth, top-k, or exotic program logic.

This is deliberately simple:
  log-mel -> adaptive pooled grid -> Linear stem -> [B, blocks, dim]
  -> sequential residual matrix layers
  -> concat final blocks -> classifier

No operation choice, no route, no attention, no class-read shortcut.
If this does not learn, the problem is data/frontend/training.  If this learns,
we use it as the stable skeleton for matrix-program experiments.
"""
from __future__ import annotations

import argparse, csv, json, math, random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from run_step_program_v3_clean_sequential import make_loaders, amp_dtype, set_seed, ensure_dir, write_json, write_jsonl

try:
    import torchaudio
except Exception:
    torchaudio = None


@dataclass
class Aux:
    layer_delta_norms: torch.Tensor   # [B,L,N]
    block_states: torch.Tensor        # [B,N,D]
    stem_norm: torch.Tensor           # [B]


class LogMelGrid(nn.Module):
    def __init__(self, sample_rate: int, n_mels: int, hop_length: int, grid_mels: int, grid_time: int):
        super().__init__()
        if torchaudio is None:
            raise RuntimeError("torchaudio required for v4 sanity baseline")
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=400, hop_length=hop_length, n_mels=n_mels, power=2.0)
        self.grid_mels = int(grid_mels)
        self.grid_time = int(grid_time)
        self.out_dim = self.grid_mels * self.grid_time

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = self.mel(wav.float().squeeze(1)).clamp_min(1e-5).log()  # [B,M,T]
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = (x - x.mean(dim=(-2, -1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            g = F.adaptive_avg_pool2d(x.unsqueeze(1), (self.grid_mels, self.grid_time)).flatten(1)
        return g


class MatrixResidualLayer(nn.Module):
    """Simple sequential matrix block.

    It has two matrix operations:
      1. block mixing across block slots, shared over channels;
      2. channel MLP per block.

    This is intentionally standard and trainable: a sanity baseline, not the final
    architecture-search system.
    """
    def __init__(self, blocks: int, dim: int, dropout: float):
        super().__init__()
        self.blocks = int(blocks)
        self.dim = int(dim)
        self.pre = nn.LayerNorm(dim)
        self.block_mix = nn.Parameter(torch.eye(blocks) + 0.02 * torch.randn(blocks, blocks))
        self.channel = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 3), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 3, dim))
        self.gate_mix = nn.Parameter(torch.tensor(-1.5))
        self.gate_channel = nn.Parameter(torch.tensor(-1.0))
        self.out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        h = self.pre(x)
        # [B,N,D] -> mix blocks: y_n = sum_m M_nm h_m
        mix = torch.einsum("bmd,nm->bnd", h, self.block_mix.to(device=x.device, dtype=x.dtype))
        ch = self.channel(h)
        delta = torch.sigmoid(self.gate_mix).to(dtype=x.dtype) * mix + torch.sigmoid(self.gate_channel).to(dtype=x.dtype) * ch
        y = self.out(x + delta)
        return y, delta.float().norm(dim=-1)


class SimpleMatrixSanityNet(nn.Module):
    def __init__(self, num_classes: int, sample_rate: int, n_mels: int, hop_length: int, grid_mels: int, grid_time: int, blocks: int, dim: int, layers: int, dropout: float):
        super().__init__()
        self.C, self.B, self.D, self.L = int(num_classes), int(blocks), int(dim), int(layers)
        self.front = LogMelGrid(sample_rate, n_mels, hop_length, grid_mels, grid_time)
        self.stem = nn.Sequential(
            nn.LayerNorm(self.front.out_dim),
            nn.Linear(self.front.out_dim, dim * blocks),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * blocks, dim * blocks),
        )
        self.block_emb = nn.Parameter(torch.randn(blocks, dim) * 0.02)
        self.layers = nn.ModuleList([MatrixResidualLayer(blocks, dim, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(dim * blocks), nn.Linear(dim * blocks, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, num_classes))

    def forward(self, wav: torch.Tensor, return_aux: bool = True):
        feat = self.front(wav)
        z = self.stem(feat).view(wav.shape[0], self.B, self.D)
        x = z + self.block_emb.to(device=z.device, dtype=z.dtype).view(1, self.B, self.D)
        deltas = []
        for layer in self.layers:
            x, dn = layer(x)
            deltas.append(dn)
        logits = self.head(x.reshape(x.shape[0], self.B * self.D))
        if not return_aux:
            return logits, None
        return logits, Aux(layer_delta_norms=torch.stack(deltas, dim=1), block_states=x, stem_norm=z.float().norm(dim=(1,2)))


def retain_aux_grads(aux: Aux):
    if aux.layer_delta_norms.requires_grad:
        aux.layer_delta_norms.retain_grad()


def make_report(model: SimpleMatrixSanityNet, aux: Aux) -> tuple[Dict, List[Dict]]:
    rows = []
    dn = aux.layer_delta_norms.detach().float().mean(0)  # [L,N]
    bs = aux.block_states.detach().float().norm(dim=-1).mean(0)  # [N]
    levels = {"layer": [], "block": []}
    for l in range(model.L):
        item = {"address": f"L{l}", "delta_norm_mean": float(dn[l].mean()), "delta_norm_max": float(dn[l].max())}
        levels["layer"].append(item); rows.append({"type": "layer", **item})
        for b in range(model.B):
            rows.append({"type": "layer_block", "address": f"L{l}.B{b}", "delta_norm": float(dn[l,b])})
    for b in range(model.B):
        item = {"address": f"B{b}", "final_state_norm": float(bs[b])}
        levels["block"].append(item); rows.append({"type": "block", **item})
    summary = {"delta_norm_mean": float(dn.mean()), "final_state_norm_mean": float(bs.mean()), "stem_norm_mean": float(aux.stem_norm.detach().float().mean())}
    return {"levels": levels, "summary": summary}, rows


def train_epoch(model, loader, opt, scaler, device, dtype, epoch, args):
    model.train(); use_amp = device.startswith("cuda") and dtype != torch.float32
    totals = {"loss":0.0,"ce":0.0,"correct":0,"n":0}; last_aux=None
    for step,(wav,y) in enumerate(loader,1):
        if args.max_train_batches and step > args.max_train_batches: break
        wav,y=wav.to(device,non_blocking=True),y.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits,aux=model(wav,return_aux=True); ce=F.cross_entropy(logits.float(),y); loss=ce
        if not torch.isfinite(loss): print("NONFINITE_LOSS skip", flush=True); continue
        scaler.scale(loss).backward()
        if args.grad_clip>0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt); scaler.update()
        bs=y.numel(); totals["loss"]+=float(loss.detach().cpu())*bs; totals["ce"]+=float(ce.detach().cpu())*bs; totals["correct"]+=int((logits.argmax(-1)==y).sum().detach().cpu()); totals["n"]+=bs
        last_aux=Aux(aux.layer_delta_norms.detach().cpu(), aux.block_states.detach().cpu(), aux.stem_norm.detach().cpu())
        if args.log_every and step % args.log_every == 0:
            print(f"epoch {epoch:03d} step {step:05d} loss={totals['loss']/max(1,totals['n']):.4f} ce={totals['ce']/max(1,totals['n']):.4f} acc={100*totals['correct']/max(1,totals['n']):.2f}%", flush=True)
    return {"loss":totals["loss"]/max(1,totals["n"]),"ce":totals["ce"]/max(1,totals["n"]),"acc":totals["correct"]/max(1,totals["n"]),"n":totals["n"],"last_aux":last_aux}


@torch.no_grad()
def evaluate(model, loader, device, dtype, args):
    model.eval(); use_amp=device.startswith("cuda") and dtype != torch.float32
    total=0.0; correct=0; n=0; last_aux=None; conf=torch.zeros(args.num_classes,args.num_classes,dtype=torch.long)
    for step,(wav,y) in enumerate(loader,1):
        if args.max_val_batches and step > args.max_val_batches: break
        wav,y=wav.to(device,non_blocking=True),y.to(device,non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits,aux=model(wav,return_aux=True); loss=F.cross_entropy(logits.float(),y)
        pred=logits.argmax(-1); bs=y.numel(); total+=float(loss.cpu())*bs; correct+=int((pred==y).sum().cpu()); n+=bs
        conf += torch.bincount((y.cpu()*args.num_classes+pred.cpu()), minlength=args.num_classes**2).view(args.num_classes,args.num_classes)
        last_aux=Aux(aux.layer_delta_norms.detach().cpu(), aux.block_states.detach().cpu(), aux.stem_norm.detach().cpu())
    return {"loss":total/max(1,n),"acc":correct/max(1,n),"n":n,"confusion":conf.tolist(),"last_aux":last_aux}


def run(args):
    set_seed(args.seed); device=args.device if args.device!="cuda" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark=True; torch.set_float32_matmul_precision("high")
    dtype=amp_dtype(args.amp); out=ensure_dir(Path(args.out_dir))
    train_loader,val_loader,classes,train_counts,val_counts=make_loaders(args); args.num_classes=len(classes)
    model=SimpleMatrixSanityNet(len(classes), args.sample_rate,args.n_mels,args.hop_length,args.grid_mels,args.grid_time,args.blocks,args.dim,args.layers,args.dropout).to(device)
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"class counts train={train_counts} val={val_counts}", flush=True)
    print(f"SimpleMatrixSanity params={sum(p.numel() for p in model.parameters())} L={args.layers} B={args.blocks} D={args.dim} grid={args.grid_mels}x{args.grid_time} device={device} amp={args.amp}", flush=True)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9,0.95))
    scaler=torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype==torch.float16)
    fields=["epoch","train_loss","train_ce","train_acc","val_loss","val_acc","best_acc"]
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=fields).writeheader()
    best=-1.0; best_epoch=0
    for epoch in range(1,args.epochs+1):
        tr=train_epoch(model,train_loader,opt,scaler,device,dtype,epoch,args); va=evaluate(model,val_loader,device,dtype,args)
        if va["acc"]>best:
            best=va["acc"]; best_epoch=epoch; torch.save({"model":model.state_dict(),"args":vars(args),"classes":classes,"best_acc":best,"epoch":epoch}, out/"best.pt")
        torch.save({"model":model.state_dict(),"args":vars(args),"classes":classes,"best_acc":best,"epoch":epoch}, out/"last.pt")
        report,rows=make_report(model,va["last_aux"])
        analysis={"epoch":epoch,"train":{k:v for k,v in tr.items() if k!="last_aux"},"val":{"loss":va["loss"],"acc":va["acc"],"n":va["n"],"confusion":va["confusion"]},"best_acc":best,"best_epoch":best_epoch,"classes":classes,"train_counts":train_counts,"val_counts":val_counts,"program_report":report}
        write_json(out/f"analysis_epoch_{epoch:03d}.json",analysis); write_jsonl(out/f"events_epoch_{epoch:03d}.jsonl",[{"epoch":epoch,**r} for r in rows])
        with (out/"metrics.csv").open("a",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=fields).writerow({"epoch":epoch,"train_loss":tr["loss"],"train_ce":tr["ce"],"train_acc":tr["acc"],"val_loss":va["loss"],"val_acc":va["acc"],"best_acc":best})
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch}", flush=True)
    write_json(out/"final_report.json", {"best_acc":best,"best_epoch":best_epoch,"args":vars(args),"classes":classes,"train_counts":train_counts,"val_counts":val_counts})


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",default="./data/speechcommands"); p.add_argument("--download",action="store_true"); p.add_argument("--synthetic",action="store_true")
    p.add_argument("--classes",default="yes,no,up,down,left,right,on,off,stop,go"); p.add_argument("--train-limit",type=int,default=12000); p.add_argument("--val-limit",type=int,default=2000)
    p.add_argument("--seconds",type=float,default=1.0); p.add_argument("--sample-rate",type=int,default=16000); p.add_argument("--n-mels",type=int,default=64); p.add_argument("--hop-length",type=int,default=160)
    p.add_argument("--grid-mels",type=int,default=32); p.add_argument("--grid-time",type=int,default=32); p.add_argument("--blocks",type=int,default=8); p.add_argument("--dim",type=int,default=128); p.add_argument("--layers",type=int,default=4); p.add_argument("--dropout",type=float,default=0.10)
    p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--eval-batch-size",type=int,default=256); p.add_argument("--workers",type=int,default=4); p.add_argument("--pin-memory",action="store_true")
    p.add_argument("--lr",type=float,default=7e-4); p.add_argument("--weight-decay",type=float,default=0.01); p.add_argument("--grad-clip",type=float,default=1.0); p.add_argument("--amp",choices=["fp16","bf16","fp32","off"],default="fp16"); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-train-batches",type=int,default=0); p.add_argument("--max-val-batches",type=int,default=0); p.add_argument("--log-every",type=int,default=50); p.add_argument("--out-dir",default="./runs/v4_simple_matrix_sanity")
    return p

if __name__ == "__main__":
    run(parser().parse_args())
