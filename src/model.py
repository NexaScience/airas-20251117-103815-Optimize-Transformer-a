"""src/model.py – ViT baseline and norm-free controller variants."""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import timm
import torch
import torch.nn as nn

__all__ = ["build_model", "load_pretrained_weights"]

###############################################################################
# Utilities -------------------------------------------------------------------
###############################################################################

def _count_nonzero(t: torch.Tensor) -> int:
    return int((t != 0).sum().item())

###############################################################################
# Controller blocks -----------------------------------------------------------
###############################################################################

class FISNBlock(nn.Module):
    """Fast Integral Scaling Normalisation – dense, norm-free."""

    def __init__(self, blk: nn.Module, dim: int, cfg):
        super().__init__()
        self.blk = blk
        self.dim = dim
        # dual variables & stats
        self.register_buffer("lam", torch.zeros(1))
        self.register_buffer("A", torch.zeros(1))
        self.tau = float(cfg.tau); self.beta = float(cfg.beta); self.eta_fast = float(cfg.eta_fast)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blk(x)
        with torch.no_grad():
            sample = out.flatten(0, 1)
            idx = torch.randint(0, sample.size(0), (min(128, sample.size(0)),), device=out.device)
            act = sample[idx].abs().mean()
            self.A.mul_(self.beta).add_(act * (1 - self.beta))
        return out  # crucial: return only tensor for timm compatibility

    def dual_loss(self) -> torch.Tensor:
        return self.lam * (self.A - self.tau)

    @torch.no_grad()
    def update(self):
        self.lam += self.eta_fast * (self.A - self.tau)
        scale = torch.exp(-self.lam)
        for m in [self.blk.attn.qkv, self.blk.attn.proj, self.blk.mlp.fc1, self.blk.mlp.fc2]:
            m.weight.data.mul_(scale)


class HISPBlock(FISNBlock):
    """Homeostatic Integral Scaling + unstructured pruning."""

    def __init__(self, blk: nn.Module, dim: int, cfg):
        super().__init__(blk, dim, cfg)
        self.theta = float(cfg.theta); self.gamma = float(cfg.gamma_prune)
        self.register_buffer("mu_hat", torch.zeros(1))

    @torch.no_grad()
    def update(self):
        super().update()
        self.mu_hat.mul_(0.99).add_(self.lam * 0.01)
        if self.mu_hat > self.theta:
            self._prune()

    def _prune(self):
        for m in [self.blk.attn.qkv, self.blk.attn.proj, self.blk.mlp.fc1, self.blk.mlp.fc2]:
            w = m.weight.data
            thr = self.gamma * w.abs().median()
            w[torch.abs(w) < thr] = 0.0
            m.weight.requires_grad = False


class MIPETBlock(nn.Module):
    """Multiscale Integral Plasticity for Energy-aware Transformers."""

    def __init__(self, blk: nn.Module, dim: int, C_flops: float, cfg):
        super().__init__()
        self.blk = blk; self.dim = dim; self.C = float(C_flops)
        # dual variables & stats
        self.register_buffer("lam", torch.zeros(1)); self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("A", torch.zeros(1)); self.register_buffer("mu_hat", torch.zeros(1))
        # hyper-parameters
        self.tau = float(cfg.tau); self.rho = float(cfg.rho); self.beta = float(cfg.beta)
        self.eta_fast = float(cfg.eta_fast); self.eta_slow = float(cfg.eta_slow)
        self.theta = float(cfg.theta); self.gamma = float(cfg.gamma_prune)

    # density helper
    def _density(self) -> float:
        nz, tot = 0, 0
        for m in [self.blk.attn.qkv, self.blk.attn.proj, self.blk.mlp.fc1, self.blk.mlp.fc2]:
            nz += _count_nonzero(m.weight)
            tot += m.weight.numel()
        return nz / max(1, tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blk(x)
        with torch.no_grad():
            sample = out.flatten(0, 1)
            idx = torch.randint(0, sample.size(0), (min(128, sample.size(0)),), device=out.device)
            act = sample[idx].abs().mean()
            self.A.mul_(self.beta).add_(act * (1 - self.beta))
        return out

    def dual_loss(self):
        density = self._density(); E_l = density * self.C
        return self.lam * (self.A - self.tau) + self.mu * (E_l - self.rho * self.C)

    @torch.no_grad()
    def update(self):
        density = self._density(); E_l = density * self.C
        self.lam += self.eta_fast * (self.A - self.tau)
        self.mu += self.eta_slow * (E_l - self.rho * self.C)
        self.mu_hat.mul_(0.99).add_(self.mu * 0.01)
        # weight fusion
        g = torch.exp(-self.lam)
        for m in [self.blk.attn.qkv, self.blk.attn.proj, self.blk.mlp.fc1, self.blk.mlp.fc2]:
            m.weight.data.mul_(g)
        # structured pruning
        if self.mu_hat > self.theta:
            self._structured_prune()

    def _structured_prune(self):
        attn = self.blk.attn; num_heads = attn.num_heads; head_dim = self.dim // num_heads
        w_qkv = attn.qkv.weight.data.view(3, num_heads, head_dim, self.dim)
        head_score = w_qkv.abs().mean(dim=(0, 2, 3))
        thr = self.gamma * head_score.median(); mask = head_score < thr
        if mask.any():
            w_qkv[:, mask] = 0.0; attn.qkv.weight.requires_grad = False
            for h in torch.nonzero(mask, as_tuple=False).flatten():
                s, e = h * head_dim, (h + 1) * head_dim
                attn.proj.weight.data[:, s:e] = 0.0; attn.proj.weight.requires_grad = False
        # FFN rows
        fc1, fc2 = self.blk.mlp.fc1, self.blk.mlp.fc2
        row_norm = fc1.weight.data.norm(dim=1)
        row_mask = row_norm < self.gamma * row_norm.median()
        if row_mask.any():
            fc1.weight.data[row_mask] = 0.0; fc1.weight.requires_grad = False
            fc2.weight.data[:, row_mask] = 0.0; fc2.weight.requires_grad = False

###############################################################################
# ViT wrappers ----------------------------------------------------------------
###############################################################################

class _ControllerViT(timm.models.vision_transformer.VisionTransformer):
    """VisionTransformer extended with integral controllers."""

    def __init__(self, meth_cfg, block_cls, **kwargs):
        super().__init__(**kwargs)
        # replace each encoder block with a wrapped one
        new_blocks: List[nn.Module] = []
        for blk in self.blocks:
            C_fl = 2 * (self.embed_dim * self.embed_dim * 3 + self.embed_dim * self.embed_dim)
            if block_cls is MIPETBlock:
                new_blocks.append(block_cls(blk, self.embed_dim, C_fl, meth_cfg))
            else:
                new_blocks.append(block_cls(blk, self.embed_dim, meth_cfg))
        self.blocks = nn.ModuleList(new_blocks)
        self._meth_cfg = meth_cfg

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        logits = super().forward(x)
        dual_loss = torch.stack([b.dual_loss() for b in self.blocks]).sum()
        return logits, {"dual_loss": dual_loss}

    # public helper --------------------------------------------------------
    def update_controllers(self):
        for b in self.blocks:
            b.update()


class FISNViT(_ControllerViT):
    def __init__(self, meth_cfg, **kwargs):
        super().__init__(meth_cfg, FISNBlock, **kwargs)


class HISPViT(_ControllerViT):
    def __init__(self, meth_cfg, **kwargs):
        super().__init__(meth_cfg, HISPBlock, **kwargs)


class MIPETViT(_ControllerViT):
    def __init__(self, meth_cfg, **kwargs):
        super().__init__(meth_cfg, MIPETBlock, **kwargs)

###############################################################################
# public factory -------------------------------------------------------------
###############################################################################

def build_model(cfg, meta):
    method = str(cfg.run.method).upper()
    name = cfg.run.model.name
    num_classes = meta["num_classes"]

    vit_kwargs = {
        "img_size": cfg.run.dataset.resolution,
        "patch_size": cfg.run.model.architecture.patch_size,
        "embed_dim": cfg.run.model.architecture.embedding_dim,
        "depth": cfg.run.model.architecture.depth,
        "num_heads": cfg.run.model.architecture.num_heads,
        "mlp_ratio": cfg.run.model.architecture.mlp_ratio,
        "num_classes": num_classes,
    }

    if method == "MIPET":
        return MIPETViT(cfg.run.method_specific, **vit_kwargs)
    elif method == "FISN":
        return FISNViT(cfg.run.method_specific, **vit_kwargs)
    elif method == "HISP":
        return HISPViT(cfg.run.method_specific, **vit_kwargs)
    else:  # baseline LayerNorm ViT
        return timm.create_model(name, pretrained=False, num_classes=num_classes)

###############################################################################
# weight loading -------------------------------------------------------------
###############################################################################

def load_pretrained_weights(model: nn.Module, weights: str):
    if weights.endswith((".pt", ".pth")) and Path(weights).exists():
        sd = torch.load(weights, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    else:
        try:
            ext = timm.create_model(weights, pretrained=True)
            model.load_state_dict(ext.state_dict(), strict=False)
        except Exception:
            pass
