"""
Transformer backbone and the GPT-QE head.

``GPT`` is a trimmed decoder-only nanoGPT backbone (Karpathy nanoGPT; GPT-2
architecture). ``GPTQE`` adds the logit-matching head of Nakaji et al. 2024:
the cumulative next-token logit is regressed onto the circuit energy, and
generation samples with a *negative*-logit Boltzmann softmax so low-energy
circuits are produced. An optional soft particle-number penalty is available
in the loss.
"""

from __future__ import annotations

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    """Trimmed decoder-only transformer backbone."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay = [p for p in param_dict.values() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = dict(fused=True) if (fused_ok and device_type == "cuda") else dict()
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)


class GPTQE(GPT):
    """Logit-matching GPT for circuit construction (Nakaji et al. 2024),
    with an optional symmetry-verification (particle-number) penalty."""

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        device = idx.device
        _, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def calculate_loss(
        self,
        tokens: torch.Tensor,
        energies: torch.Tensor,
        n_violations: Optional[torch.Tensor] = None,
        penalty_weight: float = 0.0,
    ) -> torch.Tensor:
        """Logit-matching MSE plus an optional lambda * dN^2 penalty (final step)."""
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        logits = self(current_tokens)
        mask = F.one_hot(next_tokens, num_classes=self.config.vocab_size)
        next_token_logits = (logits * mask).sum(axis=2)
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)

        target = energies
        if n_violations is not None and penalty_weight > 0.0:
            penalty_final = penalty_weight * (n_violations ** 2)
            pad = torch.zeros_like(target)
            pad[:, -1] = penalty_final
            target = target + pad
        return torch.mean(torch.square(cumsum_logits - target))

    @torch.no_grad()
    def generate(
        self, n_sequences: int, max_new_tokens: int,
        temperature: float = 1.0, device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively sample sequences with a negative-logit Boltzmann softmax."""
        idx = torch.zeros((n_sequences, 1), dtype=torch.long, device=device)
        total_logits = torch.zeros((n_sequences, 1), device=device)
        for _ in range(max_new_tokens):
            idx_cond = (idx if idx.size(1) <= self.config.block_size
                        else idx[:, -self.config.block_size:])
            logits = self(idx_cond)[:, -1, :]
            logits[:, 0] = float("inf")                 # forbid the BOS token
            probs = F.softmax(-logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            total_logits += torch.gather(logits, index=idx_next, dim=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx, total_logits
