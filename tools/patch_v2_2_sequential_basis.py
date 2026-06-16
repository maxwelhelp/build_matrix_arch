#!/usr/bin/env python3
"""Patch StepProgram v2 with sequential-basis routing.

This patch does NOT add attention, new primitives, or a plateau controller.
It only makes the existing routing more logical and less shortcut-prone:

  1. Layer route becomes a small sequential basis:
       input_skip, prev_same_block, prev_layer_mean
     Memory/global are removed from layer-level route for now.

  2. Layer route gets a strong differentiable prior:
       prev_same_block high, input_skip weak, prev_layer_mean weaker.

  3. Step read gets phase priors:
       S0 prefers state/input;
       S1 prefers prev_step;
       S2+ prefers prev_step/all_prev_steps and may use registers.

  4. Reports get sequential metrics:
       sequential_route_mass, input_skip_mass, prev_layer_mean_mass,
       step_prev_mass, step_input_mass, step_register_mass,
       identity_mass, nontrivial_mass.

The goal is to restore specialization:

  L0 extract -> L1 transform/compare -> L2 repair/aggregate -> output
"""
from pathlib import Path

path = Path("experiments/step_program/run_step_program_v2_projected_topk.py")
text = path.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Shrink layer route basis.
# ---------------------------------------------------------------------------
old = '    ROUTES = ["input", "prev_same_block", "prev_layer_mean", "global_mean", "memory_mean"]\n'
new = '    ROUTES = ["input_skip", "prev_same_block", "prev_layer_mean"]\n'
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 2. Add sequential route prior and remove global/memory from route sources.
# ---------------------------------------------------------------------------
old = '''            route_logits = self.route_net(torch.cat([state, prev_layer, glob_mean, mem_mean], dim=-1))
            route_gates = torch.softmax(route_logits.float(), dim=-1).to(dtype)
            route_src = torch.stack([base[:,None,:].expand(-1,self.N,-1), prev_layer, prev_mean, glob_mean, mem_mean], dim=2)
            route_ctx = torch.einsum("bnr,bnrd->bnd", route_gates, route_src)
'''
new = '''            route_logits = self.route_net(torch.cat([state, prev_layer, glob_mean, mem_mean], dim=-1))
            # Sequential basis prior.  The route is still differentiable, but
            # the default program is L(l-1).same_block -> L(l).same_block.
            # Order matches ROUTES: input_skip, prev_same_block, prev_layer_mean.
            route_prior = torch.tensor([-0.5, 2.0, -1.0], device=device, dtype=route_logits.dtype).view(1, 1, self.LR)
            route_gates = torch.softmax((route_logits + route_prior).float(), dim=-1).to(dtype)
            route_src = torch.stack([base[:,None,:].expand(-1,self.N,-1), prev_layer, prev_mean], dim=2)
            route_ctx = torch.einsum("bnr,bnrd->bnd", route_gates, route_src)
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 3. Step-read phase priors.
# ---------------------------------------------------------------------------
old = '''                read_logits = self.read_net(torch.cat([state, in_ctx, route_ctx, glob_ctx, mem_ctx], dim=-1))
                if s == 0:
                    read_logits = read_logits.clone(); read_logits[...,2] = -1e4; read_logits[...,3] = -1e4
                read_gates = torch.softmax((read_logits / max(1e-4, read_temp)).float(), dim=-1).to(dtype)
'''
new = '''                read_logits = self.read_net(torch.cat([state, in_ctx, route_ctx, glob_ctx, mem_ctx], dim=-1))
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
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 4. Add sequential metrics to make_reports.
# ---------------------------------------------------------------------------
old = '''    cls=aux.class_read.detach().float().mean(0); slot_use=cls.mean(0)
    for l in range(model.L):
'''
new = '''    cls=aux.class_read.detach().float().mean(0); slot_use=cls.mean(0)
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
'''
if old in text and new not in text:
    text = text.replace(old, new)

old = '''    if grad_report:
      by_addr=grad_report.get("by_address",{})
      for r in rows:
        if r["address"] in by_addr: r["grad_x_gate"] = by_addr[r["address"]]
    return {"levels":levels,"exec_stats":aux.exec_stats,"grad_report":grad_report}, rows
'''
new = '''    if grad_report:
      by_addr=grad_report.get("by_address",{})
      for r in rows:
        if r["address"] in by_addr: r["grad_x_gate"] = by_addr[r["address"]]
    return {"levels":levels,"exec_stats":aux.exec_stats,"sequential_metrics":seq_metrics,"grad_report":grad_report}, rows
'''
if old in text and new not in text:
    text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print(f"patched {path} with v2.2 sequential-basis routing")
