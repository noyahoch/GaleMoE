from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import Tensor


def default_svd_cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(base).expanduser() / "gale" / "svd_gate"


def _safe_model_id(model_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in model_id)[:240]


def gate_proj_fingerprint(expert_gate_proj_weights: list[Tensor]) -> str:
    h = hashlib.sha256()
    for w in expert_gate_proj_weights:
        t = w.detach().cpu().to(torch.float32).contiguous()
        h.update(t.numpy().tobytes())
        h.update(str(tuple(t.shape)).encode("ascii"))
    return h.hexdigest()


def _svd_gate_compute(
    expert_gate_proj_weights: list[Tensor],
    skip: int,
    k: int,
    svd_device: torch.device | None,
    *,
    residual_mean: bool = False,
) -> Tensor:
    """
    If residual_mean: each expert matrix is (W_e - mean_e' W_e') before SVD, so singular
    directions emphasize what differs from the layer-average expert, not the shared part.
    """
    compute_dev = svd_device if svd_device is not None else torch.device("cpu")
    use_cuda = compute_dev.type == "cuda"
    weights_f = [W.float().to(compute_dev, non_blocking=use_cuda) for W in expert_gate_proj_weights]
    if residual_mean and weights_f:
        W_mean = torch.stack(weights_f, dim=0).mean(dim=0)
        weights_f = [W - W_mean for W in weights_f]

    rows: list[Tensor] = []
    for W_f in weights_f:
        _, _, Vt = torch.linalg.svd(W_f, full_matrices=False)
        n = Vt.shape[0]
        lo = min(skip, n - 1)
        hi = min(skip + k, n)

        mid_vecs = Vt[lo:hi]
        direction = mid_vecs.mean(dim=0)
        direction = direction / (direction.norm() + 1e-8)
        rows.append(direction.detach().cpu())

    return torch.stack(rows, dim=0)


def svd_gate_init(
    expert_gate_proj_weights: list[Tensor],
    skip: int = 0,
    k: int = 8,
    svd_device: torch.device | None = None,
    use_cache: bool = True,
    cache_dir: Path | str | None = None,
    model_id: str = "",
    layer_idx: int = 0,
    verbose: bool = True,
    residual_mean: bool = False,
) -> Tensor:
    n_exp = len(expert_gate_proj_weights)
    fp = gate_proj_fingerprint(expert_gate_proj_weights)
    res_tag = "resmean1" if residual_mean else "resmean0"
    path: Path | None = None
    if use_cache and model_id:
        root = Path(cache_dir).expanduser() if cache_dir else default_svd_cache_root()
        path = (
            root
            / _safe_model_id(model_id)
            / f"L{layer_idx:02d}_skip{skip}_k{k}_{res_tag}_{fp}.pt"
        )
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (
                isinstance(payload, dict)
                and payload.get("fingerprint") == fp
                and "W_gate_init" in payload
            ):
                Wg = payload["W_gate_init"]
                if isinstance(Wg, Tensor) and Wg.shape[0] == n_exp:
                    if verbose:
                        print(
                            f"[gated_svd] layer {layer_idx:2d}  cache hit  ({n_exp} experts, {res_tag})",
                            flush=True,
                        )
                    return Wg.detach().cpu().float()

    dev_s = str(svd_device) if svd_device is not None else "cpu"
    if verbose:
        rm = "  residual=layer-mean" if residual_mean else ""
        print(
            f"[gated_svd] layer {layer_idx:2d}  SVD on {dev_s}  ({n_exp} experts × skip={skip} k={k}){rm} …",
            flush=True,
        )

    out = _svd_gate_compute(
        expert_gate_proj_weights, skip, k, svd_device, residual_mean=residual_mean
    )

    if use_cache and model_id and path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"fingerprint": fp, "W_gate_init": out.cpu()}, path)
        if verbose:
            print(f"[gated_svd] layer {layer_idx:2d}  saved cache  {path}", flush=True)
    elif verbose:
        print(f"[gated_svd] layer {layer_idx:2d}  SVD done (no cache write)", flush=True)

    return out
