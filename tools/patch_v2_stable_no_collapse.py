#!/usr/bin/env python3
"""Patch StepProgram v2 with stable no-collapse training support.

This patch does NOT add new primitives and does NOT add a plateau controller.
It only fixes the training/schedule failure found in the v2 projected-topk report:

  - early cost/write/entropy pressure pushed the model into identity/no-write;
  - primitive/read/transition entropy collapsed to ~0;
  - write gates collapsed to ~1e-7;
  - class_read carried most gradient while step-program wrote almost nothing.

Added mechanisms:

  1. delayed regularization for cost/write/entropy penalties;
  2. anti-collapse activity losses during warmup:
       - minimum write activity;
       - minimum nontrivial primitive mass;
       - entropy floors for primitive/read/transition gates;
  3. temporary differentiable write floor in forward pass;
  4. metrics for all new losses and effective lambdas.

The goal is a stable foundation for analysis, not extra architecture capacity.
"""
from pathlib import Path

path = Path("experiments/step_program/run_step_program_v2_projected_topk.py")
text = path.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Forward signature: add write_floor.
# ---------------------------------------------------------------------------
old = '''    def forward(self, wav: torch.Tensor, epoch: int = 1, primitive_temp: float = 1.0, read_temp: float = 1.0, primitive_exec_mode: str = "all", topk_primitives: int = 3, topk_warmup_epochs: int = 3, shadow_prob: float = 0.0, return_aux: bool = True):
'''
new = '''    def forward(self, wav: torch.Tensor, epoch: int = 1, primitive_temp: float = 1.0, read_temp: float = 1.0, primitive_exec_mode: str = "all", topk_primitives: int = 3, topk_warmup_epochs: int = 3, shadow_prob: float = 0.0, write_floor: float = 0.0, return_aux: bool = True):
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 2. Forward write gate: add differentiable floor.
# ---------------------------------------------------------------------------
old = '''                write_gate = torch.sigmoid(self.write_net(torch.cat([state, x, read, task], dim=-1))).squeeze(-1)
                write_gate = write_gate * (1.0 - 0.70 * safe_mass).clamp(0.05, 1.0)
                state = self.state_norm(state + write_gate.unsqueeze(-1) * x)
'''
new = '''                write_gate = torch.sigmoid(self.write_net(torch.cat([state, x, read, task], dim=-1))).squeeze(-1)
                write_gate = write_gate * (1.0 - 0.70 * safe_mass).clamp(0.05, 1.0)
                # Temporary anti-collapse write floor.  This is differentiable:
                # even when the learned write_gate is tiny, gradients still flow
                # through it because we use floor + (1-floor)*gate, not clamp.
                if write_floor > 0.0:
                    wf = float(max(0.0, min(0.25, write_floor)))
                    write_gate = wf + (1.0 - wf) * write_gate
                state = self.state_norm(state + write_gate.unsqueeze(-1) * x)
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 3. Replace aux_losses with delayed regularization + anti-collapse floors.
# ---------------------------------------------------------------------------
start = text.find('def aux_losses(model: StepProgramNet, aux: Aux, epoch: int, epochs: int, args):')
end = text.find('\ndef retain_aux_grads(aux: Aux):', start)
if start == -1 or end == -1:
    raise SystemExit("Could not find aux_losses block")
new_block = r'''def aux_losses(model: StepProgramNet, aux: Aux, epoch: int, epochs: int, args):
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
'''
if new_block not in text:
    text = text[:start] + new_block + text[end:]

# ---------------------------------------------------------------------------
# 4. Add new loss terms in train_epoch.
# ---------------------------------------------------------------------------
old = '''        loss=ce + losses["lambda_cost_eff"]*losses["cost_loss"] + losses["lambda_primitive_entropy_eff"]*losses["primitive_entropy"] + losses["lambda_transition_entropy_eff"]*losses["transition_entropy"] + losses["lambda_read_entropy_eff"]*losses["read_entropy"] + losses["lambda_write_l1_eff"]*losses["write_l1"]
'''
new = '''        loss=ce + losses["lambda_cost_eff"]*losses["cost_loss"] + losses["lambda_primitive_entropy_eff"]*losses["primitive_entropy"] + losses["lambda_transition_entropy_eff"]*losses["transition_entropy"] + losses["lambda_read_entropy_eff"]*losses["read_entropy"] + losses["lambda_write_l1_eff"]*losses["write_l1"]
        loss = loss + losses["lambda_write_activity_eff"] * losses["write_activity_loss"]
        loss = loss + losses["lambda_nontrivial_activity_eff"] * losses["nontrivial_activity_loss"]
        loss = loss + losses["lambda_primitive_entropy_floor_eff"] * losses["primitive_entropy_floor_loss"]
        loss = loss + losses["lambda_read_entropy_floor_eff"] * losses["read_entropy_floor_loss"]
        loss = loss + losses["lambda_transition_entropy_floor_eff"] * losses["transition_entropy_floor_loss"]
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 5. Compute/pass write_floor in train and eval.
# ---------------------------------------------------------------------------
old = '''    progress=(epoch-1)/max(1,args.epochs-1); ptemp=args.primitive_temp_start+(args.primitive_temp_end-args.primitive_temp_start)*progress; rtemp=args.read_temp_start+(args.read_temp_end-args.read_temp_start)*progress
'''
new = '''    progress=(epoch-1)/max(1,args.epochs-1); ptemp=args.primitive_temp_start+(args.primitive_temp_end-args.primitive_temp_start)*progress; rtemp=args.read_temp_start+(args.read_temp_end-args.read_temp_start)*progress
    wf_prog = min(1.0, max(0.0, (epoch - 1) / max(1, int(getattr(args, "write_floor_epochs", 6)) - 1)))
    write_floor = float(getattr(args, "write_floor_start", 0.02)) + (float(getattr(args, "write_floor_end", 0.0)) - float(getattr(args, "write_floor_start", 0.02))) * wf_prog
'''
if old in text and new not in text:
    text = text.replace(old, new)

old = '''        logits,aux=model(wav,epoch=epoch,primitive_temp=ptemp,read_temp=rtemp,primitive_exec_mode=args.primitive_exec_mode,topk_primitives=args.topk_primitives,topk_warmup_epochs=args.topk_warmup_epochs,shadow_prob=args.shadow_prob,return_aux=True)
'''
new = '''        logits,aux=model(wav,epoch=epoch,primitive_temp=ptemp,read_temp=rtemp,primitive_exec_mode=args.primitive_exec_mode,topk_primitives=args.topk_primitives,topk_warmup_epochs=args.topk_warmup_epochs,shadow_prob=args.shadow_prob,write_floor=write_floor,return_aux=True)
'''
if old in text and new not in text:
    text = text.replace(old, new)

old = '''    out={"loss":totals["loss"]/max(1,totals["n"]),"ce":totals["ce"]/max(1,totals["n"]),"acc":totals["correct"]/max(1,totals["n"]),"n":totals["n"],"last_aux":last_aux,"grad_report":grad_rep,"primitive_temp":ptemp,"read_temp":rtemp}
'''
new = '''    out={"loss":totals["loss"]/max(1,totals["n"]),"ce":totals["ce"]/max(1,totals["n"]),"acc":totals["correct"]/max(1,totals["n"]),"n":totals["n"],"last_aux":last_aux,"grad_report":grad_rep,"primitive_temp":ptemp,"read_temp":rtemp,"write_floor":write_floor}
'''
if old in text and new not in text:
    text = text.replace(old, new)

old = '''      with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp): logits,aux=model(wav,epoch=epoch,primitive_temp=args.primitive_temp_end,read_temp=args.read_temp_end,primitive_exec_mode=args.primitive_exec_mode,topk_primitives=args.topk_primitives,topk_warmup_epochs=args.topk_warmup_epochs,shadow_prob=0.0,return_aux=True); loss=F.cross_entropy(logits.float(),y)
'''
new = '''      wf_prog = min(1.0, max(0.0, (epoch - 1) / max(1, int(getattr(args, "write_floor_epochs", 6)) - 1)))
      write_floor = float(getattr(args, "write_floor_start", 0.02)) + (float(getattr(args, "write_floor_end", 0.0)) - float(getattr(args, "write_floor_start", 0.02))) * wf_prog
      with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp): logits,aux=model(wav,epoch=epoch,primitive_temp=args.primitive_temp_end,read_temp=args.read_temp_end,primitive_exec_mode=args.primitive_exec_mode,topk_primitives=args.topk_primitives,topk_warmup_epochs=args.topk_warmup_epochs,shadow_prob=0.0,write_floor=write_floor,return_aux=True); loss=F.cross_entropy(logits.float(),y)
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 6. Extend metrics fields.
# ---------------------------------------------------------------------------
old = '''    fields=["epoch","train_loss","train_ce","train_acc","val_loss","val_acc","best_acc","cost_loss","primitive_entropy","transition_entropy","read_entropy","write_l1","exec_speedup"]
'''
new = '''    fields=["epoch","train_loss","train_ce","train_acc","val_loss","val_acc","best_acc","cost_loss","primitive_entropy","transition_entropy","read_entropy","write_l1","nontrivial_mass","safe_mass","write_activity_loss","nontrivial_activity_loss","primitive_entropy_floor_loss","read_entropy_floor_loss","transition_entropy_floor_loss","lambda_cost_eff","lambda_write_l1_eff","lambda_write_activity_eff","lambda_nontrivial_activity_eff","lambda_primitive_entropy_floor_eff","lambda_read_entropy_floor_eff","lambda_transition_entropy_floor_eff","reg_w","anti_w","write_floor","exec_speedup"]
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 7. Add CLI arguments.
# ---------------------------------------------------------------------------
old = '''    p.add_argument("--lambda-cost",type=float,default=0.005); p.add_argument("--lambda-primitive-entropy",type=float,default=0.002); p.add_argument("--lambda-transition-entropy",type=float,default=0.001); p.add_argument("--lambda-read-entropy",type=float,default=0.001); p.add_argument("--lambda-write-l1",type=float,default=0.002)
    p.add_argument("--grad-analytics-every",type=int,default=50); p.add_argument("--max-train-batches",type=int,default=0); p.add_argument("--max-val-batches",type=int,default=0); p.add_argument("--log-every",type=int,default=50); p.add_argument("--out-dir",default="./runs/step_program_v2_projected_topk")
'''
new = '''    p.add_argument("--lambda-cost",type=float,default=0.005); p.add_argument("--lambda-primitive-entropy",type=float,default=0.002); p.add_argument("--lambda-transition-entropy",type=float,default=0.001); p.add_argument("--lambda-read-entropy",type=float,default=0.001); p.add_argument("--lambda-write-l1",type=float,default=0.002)
    p.add_argument("--reg-warmup-epochs",type=int,default=5)
    p.add_argument("--anti-collapse-epochs",type=int,default=6); p.add_argument("--anti-collapse-tail",type=float,default=0.25)
    p.add_argument("--lambda-write-activity",type=float,default=0.05); p.add_argument("--min-write-target",type=float,default=0.02)
    p.add_argument("--lambda-nontrivial-activity",type=float,default=0.05); p.add_argument("--min-nontrivial-mass",type=float,default=0.10)
    p.add_argument("--lambda-primitive-entropy-floor",type=float,default=0.02); p.add_argument("--primitive-entropy-floor",type=float,default=0.35)
    p.add_argument("--lambda-read-entropy-floor",type=float,default=0.01); p.add_argument("--read-entropy-floor",type=float,default=0.35)
    p.add_argument("--lambda-transition-entropy-floor",type=float,default=0.005); p.add_argument("--transition-entropy-floor",type=float,default=0.25)
    p.add_argument("--write-floor-start",type=float,default=0.02); p.add_argument("--write-floor-end",type=float,default=0.0); p.add_argument("--write-floor-epochs",type=int,default=6)
    p.add_argument("--grad-analytics-every",type=int,default=50); p.add_argument("--max-train-batches",type=int,default=0); p.add_argument("--max-val-batches",type=int,default=0); p.add_argument("--log-every",type=int,default=50); p.add_argument("--out-dir",default="./runs/step_program_v2_projected_topk")
'''
if old in text and new not in text:
    text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print(f"patched {path} with stable no-collapse support")
