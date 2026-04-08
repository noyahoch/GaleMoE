"""Load expert SVD vectors and apply vector interventions (project-out, inject, subtract)."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, Iterator

import torch

logger = logging.getLogger(__name__)


class ExpertVectors:
    """Per-expert SVD vectors loaded from pickle files.

    Cache convention: each file is a raw tensor V of shape (dim, dim) with
    columns = right singular vectors, or a dict with key "Vh" holding a matrix
    with rows = vectors.  The loader normalises both to [top_k, dim].
    """

    def __init__(
        self,
        svd_dir: str,
        layer_idx: int,
        num_experts: int = 8,
        model_tag: str = "mistralai_Mixtral_8x7B_v0.1",
        top_k: int = 1,
    ) -> None:
        self.svd_dir = svd_dir
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.model_tag = model_tag
        self.top_k = top_k
        self._vectors: Dict[int, torch.Tensor] = {}

    # -- I/O ----------------------------------------------------------------

    def load(self) -> None:
        """Load all expert vectors from svd_dir."""
        self._vectors.clear()
        logger.info(
            "Loading SVD vectors from %s (layer %d, top_k=%d)",
            os.path.abspath(self.svd_dir),
            self.layer_idx,
            self.top_k,
        )
        for i in range(self.num_experts):
            path = self._path_for(i)
            self._vectors[i] = self._load_single(path)

    def _path_for(self, expert_idx: int) -> str:
        return os.path.join(
            self.svd_dir,
            f"{self.model_tag}_layer{self.layer_idx}_expert{expert_idx}.pkl",
        )

    def _load_single(self, filepath: str) -> torch.Tensor:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"SVD file not found: {filepath}")
        with open(filepath, "rb") as f:
            obj = pickle.load(f)

        # Extract tensor from various formats
        t = self._extract_tensor(obj, filepath)
        t = torch.as_tensor(t, dtype=torch.float32, device="cpu")

        if t.ndim == 1:
            return t / (t.norm() + 1e-12)

        # 2-D: columns = vectors for square raw tensors; rows = vectors otherwise
        t = self._select_top_k(t)

        # Normalize each vector
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-12)

        return t[0] if t.shape[0] == 1 else t  # [dim] when top_k=1

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _extract_tensor(obj, filepath: str):
        """Pull out the tensor/array from the pickle object."""
        if isinstance(obj, dict):
            for key in ("Vh", "vh", "V", "v"):
                if key in obj:
                    return obj[key]
        if isinstance(obj, (tuple, list)) and len(obj) >= 3:
            return obj[2]
        if isinstance(obj, (torch.Tensor,)):
            return obj
        # numpy arrays are also acceptable (torch.as_tensor handles them)
        try:
            return torch.as_tensor(obj)
        except Exception:
            pass
        raise ValueError(f"Could not extract vector from {filepath}")

    def _select_top_k(self, t: torch.Tensor) -> torch.Tensor:
        """From a 2-D matrix, return the first top_k vectors as [k, dim].

        Square raw cache: columns = vectors → take first k columns, transpose.
        Otherwise: rows = vectors → take first k rows.
        """
        if t.shape[0] == t.shape[1] or t.shape[0] > t.shape[1]:
            # columns = vectors
            k = min(self.top_k, t.shape[1])
            return t[:, :k].T
        # rows = vectors
        k = min(self.top_k, t.shape[0])
        return t[:k]

    # -- access --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._vectors)

    def __getitem__(self, expert_idx: int) -> torch.Tensor:
        return self._vectors[expert_idx]

    def get_single(self, expert_idx: int) -> torch.Tensor:
        """First (top) vector for an expert, always 1-D [dim]."""
        v = self._vectors[expert_idx]
        return v[0] if v.ndim == 2 else v

    def keys(self) -> Iterator[int]:
        return iter(self._vectors)

    def items(self) -> Iterator[tuple[int, torch.Tensor]]:
        return iter(self._vectors.items())


# ---------------------------------------------------------------------------
# Vector interventions
# ---------------------------------------------------------------------------


def _to_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() if not x.is_floating_point() else x


class VectorIntervention:
    """Stateless vector operations for router interventions."""

    # -- direction generators ------------------------------------------------
    # Three types of random directions:
    # 1. make_random: Single random direction in full R^dim
    # 2. make_random_subspace: k orthonormal random directions in full R^dim
    # 3. make_random_in_span: Single random direction in span of given vectors

    @staticmethod
    def make_random(dim: int, seed: int) -> torch.Tensor:
        """Random unit vector in R^dim (uniformly distributed on unit sphere)."""
        g = torch.Generator().manual_seed(seed)
        r = torch.randn(dim, generator=g)
        return r / (r.norm() + 1e-12)

    @staticmethod
    def make_random_subspace(
        k: int, dim: int, seed: int, device: torch.device = None
    ) -> torch.Tensor:
        """Create k random orthonormal vectors in R^dim using QR decomposition.

        Useful for testing interventions with completely random subspaces
        (not constrained to SVD span).

        Args:
            k: Number of orthonormal vectors to generate.
            dim: Dimension of the ambient space.
            seed: Random seed for reproducibility.
            device: Device to create tensors on (default: cpu).

        Returns:
            [k, dim] tensor of k orthonormal random vectors.
        """
        if device is None:
            device = torch.device("cpu")

        # Generate random matrix
        g = torch.Generator(device=device).manual_seed(seed)
        mat = torch.randn(dim, k, generator=g, device=device, dtype=torch.float32)

        # QR decomposition gives orthonormal columns
        q, r = torch.linalg.qr(mat)

        # Return first k columns as rows: [k, dim]
        return q.T[:k]

    @staticmethod
    def make_random_in_span(vectors: torch.Tensor, seed: int) -> torch.Tensor:
        """Random unit vector in the span of *vectors* (constrained to SVD subspace).

        Args:
            vectors: [k, dim] (rows = basis) or [dim] (single direction).
            seed: Random seed.

        Returns:
            [dim] random unit vector in span(vectors).

        Note:
            Unlike make_random_subspace which creates k orthonormal vectors,
            this creates 1 random direction within the given subspace.
            This is the "random" baseline used in experiments to test if the
            specific SVD directions matter vs. any random direction in the same span.
        """
        g = torch.Generator().manual_seed(seed)
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        k, dim = vectors.shape
        coeffs = torch.randn(k, generator=g, dtype=vectors.dtype, device=vectors.device)
        r = (coeffs.unsqueeze(0) @ vectors).squeeze(0)
        return r / (r.norm() + 1e-12)

    @staticmethod
    def make_orthogonal(v: torch.Tensor, seed: int) -> torch.Tensor:
        """Random unit vector orthogonal to *v* (Gram–Schmidt)."""
        g = torch.Generator().manual_seed(seed)
        r = torch.randn(v.shape, generator=g, dtype=v.dtype, device=v.device)
        r = r - torch.dot(r, v) * v
        return r / (r.norm() + 1e-12)

    # -- interventions -------------------------------------------------------

    @staticmethod
    def project_out(
        base: torch.Tensor, remove: torch.Tensor, scale: float = 1.0
    ) -> torch.Tensor:
        """Remove the component(s) of *base* parallel to *remove*.

        Args:
            base: [dim] vector to modify.
            remove: [dim] single direction **or** [k, dim] subspace basis.
            scale: Amplification factor for projection (1.0=normal, >1=amplified ablation).

        Returns:
            [dim] projected vector (same dtype as *base*).
        """
        orig_dtype = base.dtype
        base = _to_float(base)
        remove = remove.to(device=base.device, dtype=base.dtype)

        if remove.ndim == 1:
            projection = torch.dot(base, remove) * remove
            result = base - scale * projection
        elif remove.ndim == 2:
            result = _project_out_subspace(base, remove, scale)
        else:
            raise ValueError(f"remove must be 1-D or 2-D, got {remove.ndim}-D")

        return result.to(orig_dtype)

    @staticmethod
    def inject(
        base: torch.Tensor,
        direction: torch.Tensor,
        scale: float = 1.0,
        *,
        weight_decay: float = 1.0,
    ) -> torch.Tensor:
        """Inject direction(s) into base.

        Args:
            base: [dim] vector to modify.
            direction: [dim] single direction **or** [k, dim] subspace.
            scale: Injection strength.
            weight_decay: When direction is [k, dim], w_i ∝ 1/(i+1)^weight_decay.

        Returns:
            [dim] modified vector.
        """
        orig_dtype = base.dtype
        base, direction = _to_float(base), _to_float(direction.to(base.device))

        if direction.ndim == 1:
            result = base + scale * direction / (direction.norm() + 1e-12)
        elif direction.ndim == 2:
            result = _inject_subspace(base, direction, scale, weight_decay=weight_decay)
        else:
            raise ValueError(f"direction must be 1-D or 2-D, got {direction.ndim}-D")

        return result.to(orig_dtype)

    @staticmethod
    def weighted_subspace_row(
        vectors: torch.Tensor,
        scale: float = 1.0,
        weight_decay: float = 1.0,
    ) -> torch.Tensor:
        """Return a single row that lies in the weighted subspace: scale * unit(weighted_sum(vectors)).
        Use to replace a router row with this direction (e.g. per-expert SVD replacement).
        vectors: [k, dim] orthonormal (e.g. top-k SVD rows). weight_decay: w_i ∝ 1/(i+1)^weight_decay.
        """
        return _weighted_subspace_row(vectors, scale, weight_decay)

    @staticmethod
    def subtract(
        base: torch.Tensor, direction: torch.Tensor, scale: float = 1.0
    ) -> torch.Tensor:
        """Subtract direction(s) from base: base − scale * unit(direction).

        Args:
            base: [dim] vector to modify.
            direction: [dim] single direction **or** [k, dim] subspace.
            scale: Subtraction strength.

        Returns:
            [dim] modified vector.
        """
        orig_dtype = base.dtype
        base, direction = _to_float(base), _to_float(direction.to(base.device))

        if direction.ndim == 1:
            # Single direction: subtract normalized vector
            result = base - scale * direction / (direction.norm() + 1e-12)
        elif direction.ndim == 2:
            # Subspace: subtract mean direction of orthogonalized vectors
            result = _subtract_subspace(base, direction, scale)
        else:
            raise ValueError(f"direction must be 1-D or 2-D, got {direction.ndim}-D")

        return result.to(orig_dtype)


def _project_out_subspace(
    base: torch.Tensor, vectors: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """Project *base* out of the subspace spanned by *vectors* [k, dim].

    Assumes vectors are orthonormal (as they are from SVD). Uses direct projection
    formula: base - scale * Σᵢ <base, vᵢ> vᵢ

    Args:
        base: [dim] vector to project.
        vectors: [k, dim] orthonormal subspace basis (e.g., from SVD).
        scale: Amplification factor for projection.

    Returns:
        [dim] projected vector.
    """
    # Compute all dot products at once: [k] = [k, dim] @ [dim]
    dots = torch.mv(vectors, base)  # <vᵢ, base> for all i

    # Compute total projection: Σᵢ <base, vᵢ> · vᵢ
    # dots: [k], vectors: [k, dim] -> sum over k -> [dim]
    subspace_projection = torch.sum(dots.unsqueeze(1) * vectors, dim=0)

    # Remove projection from base
    return base - scale * subspace_projection


def _weighted_subspace_row(
    vectors: torch.Tensor,
    scale: float = 1.0,
    weight_decay: float = 1.0,
) -> torch.Tensor:
    """Return scale * unit(weighted_sum(vectors)) for replacing a router row. vectors: [k, dim]."""
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D [k,d], got {tuple(vectors.shape)}")
    k, d = vectors.shape
    orig_dtype = vectors.dtype
    vectors = vectors.float()
    eps = 1e-12
    idx = torch.arange(1, k + 1, device=vectors.device, dtype=vectors.dtype)
    w = 1.0 / (idx**weight_decay)
    w = w / (w.sum() + eps)
    delta = (w[:, None] * vectors).sum(dim=0)
    unit = delta / (delta.norm() + eps)
    return (scale * unit).to(orig_dtype)


def _inject_subspace(
    base: torch.Tensor,
    vectors: torch.Tensor,
    scale: float = 1.0,
    weight_decay: float = 1.0,
) -> torch.Tensor:
    """Inject the weighted subspace direction into `base`.

    Uses the same weighted combination as _weighted_subspace_row: base + scale * unit(weighted_sum(vectors)),
    with w_i ∝ 1/(i+1)^weight_decay (earlier vectors get larger weight).

    Assumes `vectors` are orthonormal (e.g., from SVD), shape [k, d].

    Args:
        base: [d]
        vectors: [k, d] (orthonormal rows)
        scale: injection strength
        weight_decay: w_i ∝ 1 / (i+1)^weight_decay (i=1..k)

    Returns:
        [d]
    """
    if base.ndim != 1:
        raise ValueError(f"`base` must be 1D [d], got {tuple(base.shape)}")
    if vectors.ndim != 2:
        raise ValueError(f"`vectors` must be 2D [k,d], got {tuple(vectors.shape)}")
    k, d = vectors.shape
    if base.shape[0] != d:
        raise ValueError(f"Dim mismatch: base has {base.shape[0]}, vectors have d={d}")

    orig_dtype = base.dtype
    base = base.float()
    vectors = vectors.float()
    direction = _weighted_subspace_row(vectors, scale=scale, weight_decay=weight_decay)
    return (base + direction).to(orig_dtype)


def _subtract_subspace(
    base: torch.Tensor, vectors: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """Subtract the mean direction of a subspace from base.

    Assumes vectors are orthonormal (as they are from SVD). Computes the mean
    of all vectors and subtracts along that direction.

    Args:
        base: [dim] vector to modify.
        vectors: [k, dim] orthonormal subspace basis (e.g., from SVD).
        scale: Subtraction strength.

    Returns:
        [dim] base - scale * unit(mean_direction)
    """
    # Compute mean direction (already orthonormal, so just average)
    mean_direction = vectors.mean(dim=0)  # [dim]

    # Normalize and subtract
    mean_direction = mean_direction / (mean_direction.norm() + 1e-12)
    return base - scale * mean_direction
