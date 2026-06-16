#!/usr/bin/env python3
"""Patch v3 and v3.1 with natural specialization fixes.

Problem confirmed by v3/v3.1 reports:
  - balanced data, but model stayed at 10% / ln(10);
  - operations were computed and update_norm grew;
  - blocks started from the same mean(evidence) + block_emb;
  - final head used mean(blocks), destroying block roles.

This patch fixes both active experiment files without adding routers/attention/top-k:
  1. block-specific input initialization from evidence via learned block queries;
  2. stronger fixed step input context schedule: S0 strong, S1 medium, S2 weaker;
  3. final concat head over all final blocks instead of mean(blocks).

The structure remains sequential and differentiable.
"""
from pathlib import Path

FILES = [
    Path("experiments/step_program/run_step_program_v3_clean_sequential.py"),
    Path("experiments/step_program/run_step_program_v3_1_no_router_matrix_chain.py"),
]


def patch_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    orig = text

    # Add block-specific evidence queries after block_emb.
    old = '        self.block_emb = nn.Parameter(torch.randn(self.B, dim) * 0.03)\n'
    new = (
        '        self.block_emb = nn.Parameter(torch.randn(self.B, dim) * 0.03)\n'
        '        # Block-specific evidence initialization.  This is not a router: every block\n'
        '        # always receives its own learned projection of the input evidence.\n'
        '        self.block_query = nn.Parameter(torch.randn(self.B, dim) * 0.03)\n'
        '        self.block_in_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))\n'
    )
    if old in text and "self.block_query" not in text:
        text = text.replace(old, new)

    # Make context_scale less tiny by default.
    text = text.replace('self.context_scale = nn.Parameter(torch.full((self.L, self.S), -2.5))', 'self.context_scale = nn.Parameter(torch.full((self.L, self.S), -1.0))')

    # Replace old head dim with concat-block head.
    old = '        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, num_classes))\n'
    new = (
        '        final_dim = dim * self.B\n'
        '        self.head = nn.Sequential(nn.LayerNorm(final_dim), nn.Linear(final_dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, num_classes))\n'
    )
    if old in text and "final_dim = dim * self.B" not in text:
        text = text.replace(old, new)

    # Add helper method after _input_context in both classes.
    old = '''    def _input_context(self, h: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:\n        score = torch.einsum("bnd,bed->bne", self.input_q(h), self.input_k(evidence)) / math.sqrt(h.shape[-1])\n        a = torch.softmax(score.float(), dim=-1).to(h.dtype)\n        return torch.einsum("bne,bed->bnd", a, self.input_v(evidence))\n\n'''
    new = '''    def _input_context(self, h: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:\n        score = torch.einsum("bnd,bed->bne", self.input_q(h), self.input_k(evidence)) / math.sqrt(h.shape[-1])\n        a = torch.softmax(score.float(), dim=-1).to(h.dtype)\n        return torch.einsum("bne,bed->bnd", a, self.input_v(evidence))\n\n    def _block_input(self, evidence: torch.Tensor) -> torch.Tensor:\n        # [B,E,D] evidence -> [B,Blocks,D] block-specific start states.\n        q = self.block_query.to(device=evidence.device, dtype=evidence.dtype)\n        score = torch.einsum("nd,bed->bne", q, evidence) / math.sqrt(evidence.shape[-1])\n        a = torch.softmax(score.float(), dim=-1).to(evidence.dtype)\n        ctx = torch.einsum("bne,bed->bnd", a, evidence)\n        return self.block_in_proj(ctx) + self.block_emb.to(device=evidence.device, dtype=evidence.dtype).view(1, self.B, self.D)\n\n'''
    if old in text and "def _block_input" not in text:
        text = text.replace(old, new)

    # Replace mean(evidence) block init in both files.  v3 has separate base variable for reports.
    old = '''        evidence = self.evidence(wav)\n        base = self.base_proj(evidence.mean(dim=1))\n        base_blocks = base[:, None, :] + self.block_emb.to(device=device, dtype=dtype).view(1, self.B, D)\n        h = self.layer_norm(base_blocks)\n'''
    new = '''        evidence = self.evidence(wav)\n        base_blocks = self._block_input(evidence)\n        base = base_blocks.mean(dim=1)\n        h = self.layer_norm(base_blocks)\n'''
    if old in text:
        text = text.replace(old, new)

    old = '''        evidence = self.evidence(wav)\n        base = self.base_proj(evidence.mean(dim=1))\n        base_blocks = base[:, None, :] + self.block_emb.to(device=device, dtype=dtype).view(1, self.B, D)\n        h = self.norm(base_blocks)\n'''
    new = '''        evidence = self.evidence(wav)\n        base_blocks = self._block_input(evidence)\n        base = base_blocks.mean(dim=1)\n        h = self.norm(base_blocks)\n'''
    if old in text:
        text = text.replace(old, new)

    # Replace weak learned-only context with fixed phase + small learned modulation.
    old = '''                ctx_gate = torch.sigmoid(self.context_scale[l, s]).to(device=device, dtype=dtype)\n                context = ctx_gate * inp_ctx + 0.10 * task\n'''
    new = '''                ctx_gate = torch.sigmoid(self.context_scale[l, s]).to(device=device, dtype=dtype)\n                # Fixed phase context: S0 strongly sees input, S1 medium, S2+ weaker.\n                # This is not routing; it is a sequential prior for audio evidence flow.\n                phase_gain = torch.tensor(1.0 / float(s + 1), device=device, dtype=dtype)\n                context = (phase_gain + 0.25 * ctx_gate) * inp_ctx + 0.10 * task\n'''
    if old in text:
        text = text.replace(old, new)

    # Replace final mean head with concat head.
    text = text.replace('        final = self.final_norm(h.mean(dim=1))\n        logits = self.head(final)\n', '        final = h.reshape(B, self.B * D)\n        logits = self.head(final)\n')
    text = text.replace('        final = self.norm(h.mean(dim=1))\n        logits = self.head(final)\n', '        final = h.reshape(B, self.B * D)\n        logits = self.head(final)\n')

    if text != orig:
        path.write_text(text, encoding="utf-8")
        print(f"patched {path}")
    else:
        print(f"already patched or no changes: {path}")


def main():
    for p in FILES:
        if not p.exists():
            raise SystemExit(f"missing file: {p}")
        patch_file(p)


if __name__ == "__main__":
    main()
