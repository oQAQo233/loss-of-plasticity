"""
Neural collapse metrics

This module contains functions computing the metrics (NC1-NC4) introduced in Papyan et al. (2020).
The code was adapted from Zhu et al. (2021): https://github.com/tding1/Neural-Collapse
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional


# ------------------------------
# Helpers
# ------------------------------

def _rebatch_loader(loader: DataLoader, max_bs: int = 128) -> DataLoader:
    """
    Rebuild a dataloader with a (possibly) smaller batch size to reduce GPU peak memory.
    Keeps dataset/shuffle/drop_last/pin_memory settings if available.
    """
    bs = getattr(loader, "batch_size", None)
    if bs is None or bs > max_bs:
        return DataLoader(
            loader.dataset,
            batch_size=max_bs,
            shuffle=getattr(loader, "shuffle", False),
            drop_last=getattr(loader, "drop_last", False),
            num_workers=getattr(loader, "num_workers", 0),
            pin_memory=getattr(loader, "pin_memory", False),
        )
    return loader


@torch.no_grad()


def _extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    h = x.to(device)  # 直接用原始输入
    
    # 按模型真实层级顺序前向
    h = model.layers[0](h)
    if len(model.layers) > 1 and isinstance(model.layers[1], nn.ReLU):
        h = model.layers[1](h)
    h = model.layers[2](h)
    if len(model.layers) > 3 and isinstance(model.layers[3], nn.ReLU):
        h = model.layers[3](h)
    h = model.layers[4](h)
    if len(model.layers) > 5 and isinstance(model.layers[5], nn.ReLU):
        h = model.layers[5](h)
    
    return h


# ------------------------------
# 新增：分类器权重提取函数（适配DeepFFNN）
# ------------------------------
@torch.no_grad()
def _get_classifier_weights(model: nn.Module) -> torch.Tensor:
    """
    适配DeepFFNN模型：提取输出层（layers.6）的权重
    确保权重维度为 [num_classes, feature_dim]
    """
    # 优先直接取输出层权重（从你的CSV保存代码可知输出层是layers.6）
    try:
        output_layer = model.layers[6]
        W = output_layer.weight.data.clone()
    except (IndexError, AttributeError):
        # 备用方案：找最后一个Linear层
        linear_layers = [m for _, m in model.named_modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise ValueError("Model has no Linear layer for classification!")
        W = linear_layers[-1].weight.data.clone()
    
    # 验证权重维度（num_classes=10）
    assert W.shape[0] == 10, f"输出层权重维度错误，期望[10, D]，实际{W.shape}"
    return W


# ------------------------------
# Means (single pass, no repeated large allocs)
# ------------------------------

@torch.no_grad()
def _get_feature_means(
    model: nn.Module,
    data_loader: DataLoader,
    num_classes: int,
    use_cache: bool = False
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Return global mean and dict of per-class means on the model's device.

    Memory-friendly implementation:
    - Do NOT allocate zeros per missing class in each batch.
    - Accumulate sums + counts, then divide once at the end.
    """
    if use_cache:
        if hasattr(_get_feature_means, "cached_mu_G") and hasattr(_get_feature_means, "cached_mu_c_dict"):
            return _get_feature_means.cached_mu_G, _get_feature_means.cached_mu_c_dict

    device = next(model.parameters()).device
    model.eval()

    # First batch to infer feature dimension D
    first_inputs, first_targets = next(iter(data_loader))
    first_inputs = first_inputs.to(device, non_blocking=True)
    first_feats = _extract_features(model, first_inputs)
    D = first_feats.shape[1]

    mu_sum = torch.zeros(D, device=device, dtype=first_feats.dtype)
    class_sum = torch.zeros(num_classes, D, device=device, dtype=first_feats.dtype)
    class_cnt = torch.zeros(num_classes, device=device, dtype=torch.long)

    # process the first batch
    mu_sum += first_feats.sum(dim=0)
    t = first_targets.to(device, non_blocking=True).long()
    for c in range(num_classes):
        mask = (t == c)
        if mask.any():
            class_sum[c] += first_feats[mask].sum(dim=0)
            class_cnt[c] += mask.sum()

    # remaining batches
    for inputs, targets in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        feats = _extract_features(model, inputs)

        mu_sum += feats.sum(dim=0)
        for c in range(num_classes):
            m = (targets == c)
            if m.any():
                class_sum[c] += feats[m].sum(dim=0)
                class_cnt[c] += m.sum()

        del feats, inputs, targets
        if device.type == "cuda":
            torch.cuda.empty_cache()

    N = len(data_loader.dataset)
    mu_G = mu_sum / max(N, 1)

    mu_c_dict: Dict[int, torch.Tensor] = {}
    for c in range(num_classes):
        count = class_cnt[c].item()
        if count > 0:
            mu_c_dict[c] = class_sum[c] / count
        else:
            mu_c_dict[c] = torch.zeros(D, device=device, dtype=class_sum.dtype)

    # cache
    _get_feature_means.cached_mu_G = mu_G
    _get_feature_means.cached_mu_c_dict = mu_c_dict

    return mu_G, mu_c_dict


# ------------------------------
# NC1: Within / Between covariance (two-pass, batchwise)
# ------------------------------

@torch.no_grad()
def NC1(
    model: nn.Module,
    num_classes: int,
    inputs: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
    use_cache: bool = False
) -> float:
    """
    Cross-example within-class variability: Tr(Sigma_W Sigma_B^+)/(K)

    - Two passes:
      (1) means (mu_c, mu_G)
      (2) Sigma_W batchwise accumulation with (X - mu_c)^T (X - mu_c)
    - No precision reduction, all in float32 on GPU.
    """
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    device = next(model.parameters()).device
    model.eval()

    # rebatch smaller for NC computation to reduce peak memory
    loader = _rebatch_loader(data_loader, max_bs=128)

    # pass 1: means
    mu_G, mu_c_dict = _get_feature_means(model=model, data_loader=loader, num_classes=num_classes, use_cache=use_cache)
    D = mu_G.shape[0]

    # pass 2: Sigma_W (classwise)
    Sigma_W = torch.zeros(D, D, device=device, dtype=mu_G.dtype)
    N = len(loader.dataset)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        feats = _extract_features(model, x)  # [B, D]

        # group by class and accumulate in blocks
        for c in range(num_classes):
            m = (y == c)
            if m.any():
                diff = feats[m] - mu_c_dict[c]  # [n_c, D]
                Sigma_W += diff.transpose(0, 1) @ diff  # [D, D]
        del feats, x, y, diff, m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    Sigma_W = Sigma_W / max(N, 1)

    # Sigma_B
    Sigma_B = torch.zeros(D, D, device=device, dtype=mu_G.dtype)
    for c in range(num_classes):
        d = (mu_c_dict[c] - mu_G).unsqueeze(1)  # [D,1]
        Sigma_B += d @ d.transpose(0, 1)
    Sigma_B = Sigma_B / max(num_classes, 1)

    # NC1
    pinv_SB = torch.linalg.pinv(Sigma_B)
    val = torch.trace(Sigma_W @ pinv_SB) / num_classes
    return val.item()


# ------------------------------
# NC2: Simplex ETF convergence (class means-based, align with paper)
# ------------------------------

@torch.no_grad()
def NC2(
    model: nn.Module,
    num_classes: int,
    inputs: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
    use_cache: bool = False
) -> float:
    """
    Align with Papyan et al. (2020) NC2 definition:
    Measure convergence of centered class means to simplex ETF via two core indicators:
    1. Coefficient of variation of class mean norms (equinorm property)
    2. Standard deviation of inter-class mean cosines (equiangular property)
    Return combined score (smaller = better NC2 convergence)
    """
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    device = next(model.parameters()).device
    loader = _rebatch_loader(data_loader, max_bs=128)

    # Get centered class means (core object of paper's NC2)
    mu_G, mu_c_dict = _get_feature_means(model=model, data_loader=loader, num_classes=num_classes, use_cache=use_cache)
    mu_centered = torch.stack([mu_c_dict[c] - mu_G for c in range(num_classes)], dim=0).to(device)  # [K, D]

    # 1. Equinorm indicator: Coefficient of variation (std / avg) of class mean norms
    norms = torch.norm(mu_centered, p=2, dim=1)  # [K]
    avg_norm = torch.mean(norms)
    std_norm = torch.std(norms)
    cv_norm = std_norm / (avg_norm + 1e-8)  # Avoid division by zero

    # 2. Equiangular indicator: Std of inter-class cosine similarities
    mu_normalized = mu_centered / (norms.unsqueeze(1) + 1e-8)  # Normalize to unit norm
    cos_matrix = mu_normalized @ mu_normalized.T  # [K, K]
    # Extract off-diagonal elements (c != c')
    off_diag_mask = torch.triu(torch.ones_like(cos_matrix, dtype=bool), diagonal=1)
    cos_vals = cos_matrix[off_diag_mask]
    std_cos = torch.std(cos_vals) if len(cos_vals) > 0 else torch.tensor(0.0, device=device)

    # Combine indicators (consistent with paper's dual conditions, smaller = better)
    nc2_score = torch.sqrt(cv_norm ** 2 + std_cos ** 2).item()
    return nc2_score


# ------------------------------
# NC3: Duality (修正为原版论文逻辑)
# ------------------------------
@torch.no_grad()
def NC3(
    model: nn.Module,
    num_classes: int,
    inputs: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
    use_cache: bool = False
) -> float:
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    device = next(model.parameters()).device
    loader = _rebatch_loader(data_loader, max_bs=128)

    # 1. 获取均值（保持你原逻辑）
    mu_G, mu_c_dict = _get_feature_means(
        model=model,
        data_loader=loader,
        num_classes=num_classes,
        use_cache=use_cache
    )

    # 2. 分类器权重 W: [K, D]
    W = _get_classifier_weights(model).to(device)

    # 3. 构建中心化类均值（保持你原逻辑）
    M = torch.stack([mu_c_dict[i] - mu_G for i in range(num_classes)], dim=0).to(device)  # [K, D]

    # =========================
    # ⭐ 核心修改从这里开始
    # =========================

    eps = 1e-8

    # 4. 逐类归一化
    W_norm = W / (torch.norm(W, dim=1, keepdim=True) + eps)
    M_norm = M / (torch.norm(M, dim=1, keepdim=True) + eps)

    # 5. 逐类 cosine
    cos_sim = torch.sum(W_norm * M_norm, dim=1)  # [K]

    # 6. NC3_k = 1 - cos
    nc3_per_class = 1.0 - cos_sim

    # 7. 平均
    nc3_score = nc3_per_class.mean().item()

    return nc3_score


# ------------------------------
# NC4: Agreement with nearest class center (vectorized on GPU)
# ------------------------------

@torch.no_grad()
def NC4(
    model: nn.Module,
    num_classes: int,
    inputs: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
    use_cache: bool = False
) -> float:
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    device = next(model.parameters()).device
    model.eval()

    loader = _rebatch_loader(data_loader, max_bs=128)
    _, mu_c_dict = _get_feature_means(model=model, data_loader=loader, num_classes=num_classes, use_cache=use_cache)

    centers = torch.stack([mu_c_dict[i] for i in range(num_classes)], dim=0).to(device)  # [K, D]
    centers_sq = (centers ** 2).sum(dim=1, keepdim=True)  # [K,1]

    agree = 0
    total = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)

        feats = _extract_features(model, x)  # [B, D]
        logits = model(x)                    # [B, K]

        # nearest center (squared Euclidean)
        # dist^2(x, c) = ||x||^2 - 2 x @ c^T + ||c||^2
        x_sq = (feats ** 2).sum(dim=1, keepdim=True)       # [B,1]
        xc = feats @ centers.T                              # [B,K]
        d2 = x_sq - 2 * xc + centers_sq.T                   # [B,K]
        nn_center = torch.argmin(d2, dim=1)                 # [B]

        pred = torch.argmax(logits, dim=1)                  # [B]
        agree += (nn_center == pred).sum().item()
        total += x.shape[0]

        del feats, logits, x, x_sq, xc, d2, nn_center, pred
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return float(agree / max(total, 1))


# ------------------------------
# NC wrapper (修正缓存逻辑+强制eval模式)
# ------------------------------

@torch.no_grad()
def NC(
    model: nn.Module,
    num_classes: int,
    inputs: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    data_loader: Optional[DataLoader] = None,
    use_cache: bool = False  # 新增参数，统一控制缓存
) -> Tuple[float, float, float, float]:
    """
    Compute all NC metrics with GPU-friendly memory usage (batchwise, two-pass).
    """
    # 关键修复：强制模型切评估模式，确保特征提取稳定
    model.eval()
    
    if data_loader is None:
        assert inputs is not None and targets is not None, "no data provided"
        data_loader = DataLoader(list(zip(inputs, targets)), batch_size=len(targets))

    # use small batchsize for NC computation to reduce peak memory (keep float32 precision)
    loader = _rebatch_loader(data_loader, max_bs=128)

    # 统一用相同的use_cache（训练时必须设为False）
    nc1 = NC1(model=model, data_loader=loader, num_classes=num_classes, use_cache=use_cache)
    nc2 = NC2(model=model, data_loader=loader, num_classes=num_classes, use_cache=use_cache)
    nc3 = NC3(model=model, data_loader=loader, num_classes=num_classes, use_cache=use_cache)
    nc4 = NC4(model=model, data_loader=loader, num_classes=num_classes, use_cache=use_cache)

    return nc1, nc2, nc3, nc4