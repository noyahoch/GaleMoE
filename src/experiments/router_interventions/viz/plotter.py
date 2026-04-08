"""Plotting for router intervention results.

Helpers
-------
load_results       – load a single JSON result file
load_results_dir   – load all result JSONs from a folder

Plots
-----
plot_delta_vs_k      – loss delta vs k for one layer (one line per variant)
plot_delta_vs_layers – loss delta vs layer for one k (one line per variant)
plot_ablation_results – bar chart of loss / delta for a single run

Publication Settings
-------------------
set_publication_style()  – Configure matplotlib for publication-quality plots
set_acl_style(use_tex=False) – ACL: 10pt serif, column widths (3.25"/6.75"), PDF font embedding.
                               use_tex=True: LaTeX-rendered text (requires pdflatex).
plot_delta_vs_k_acl(), plot_delta_vs_layers_acl() – Save as PDF (vector, embedded fonts).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np

# PDF: embed fonts so text stays crisp and matches paper (ACL-friendly)
PDF_EMBED_FONTS = {
    'pdf.fonttype': 42,  # TrueType embedded in PDF (vector, not outline)
    'ps.fonttype': 42,
}

# ACL / publication: lighter, elegant palette (colorblind-friendly, prints well in B&W)
ACL_COLORS = ['#6BAED6', '#E6A23C', '#2E9D8F', '#C77EB5', '#CA9161', '#949494']
ACL_MARKERS = ['o', 's', '^', 'D', 'v', 'P']

# Lighter Blues for heatmaps: bright (white) low end, avoids very dark blue at high end
_LIGHT_BLUES = mpl_colors.LinearSegmentedColormap.from_list(
    'light_blues', plt.cm.Blues(np.linspace(0.0, 0.75, 256))
)

# Publication-quality settings
PUBLICATION_SETTINGS = {
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    **PDF_EMBED_FONTS,
}

def set_publication_style():
    """Apply publication-quality matplotlib settings."""
    plt.rcParams.update(PUBLICATION_SETTINGS)

def set_acl_style(use_tex: bool = False):
    """Apply ACL conference paper specific matplotlib settings.
    
    ACL prefers figures that match the paper: PDF with embedded fonts (vector),
    and optionally LaTeX-rendered text so labels/ticks match the paper font.
    
    Figure widths:
    - Single column: 3.25 inches
    - Double column: 6.75 inches
    
    Args:
        use_tex: If True, use LaTeX for all text (requires pdflatex). If False,
            use serif fonts (Times/DejaVu Serif) and still export vector PDF with
            embedded fonts.
    """
    acl_settings = {
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'figure.constrained_layout.use': True,
        **PDF_EMBED_FONTS,
    }
    if use_tex:
        acl_settings['text.usetex'] = True
        # Match paper font (Times-like); fallback to default if mathptmx missing
        acl_settings['text.latex.preamble'] = r'\usepackage{mathptmx}'
    else:
        acl_settings['text.usetex'] = False
    plt.rcParams.update(acl_settings)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_results(filepath: str | Path) -> Dict[str, Any]:
    """Load a single results JSON."""
    with open(filepath) as f:
        data = json.load(f)
    
    # Backward compatibility: if using old format (k-independent stored redundantly),
    # normalize to new format with k_independent section.
    # Only zero, shuffle, orthogonal are k-independent; random is per-k in by_k.
    if "k_independent" not in data and "by_k" in data:
        first_k = next(iter(data["by_k"].values()), {})
        k_independent = {}
        for variant in ["zero", "shuffle", "orthogonal"]:
            if variant in first_k:
                k_independent[variant] = first_k[variant]
        if k_independent:
            data["k_independent"] = k_independent
            for k_entry in data["by_k"].values():
                for variant in k_independent:
                    k_entry.pop(variant, None)
    
    return data


def load_results_dir(results_dir: str | Path, glob: str = "project_out_L*.json") -> List[Dict[str, Any]]:
    """Load all matching JSONs from *results_dir*, sorted by layer index."""
    paths = sorted(Path(results_dir).glob(glob))
    if not paths:
        raise FileNotFoundError(f"No files matching {glob!r} in {results_dir}")
    out = [load_results(p) for p in paths]
    out.sort(key=lambda r: r.get("config", {}).get("layer_idx", 0))
    return out


# ---------------------------------------------------------------------------
# delta vs k  (single layer, multiple k)
# ---------------------------------------------------------------------------

def plot_delta_vs_k(
    results: Dict[str, Any],
    *,
    variants: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 5),
    ylim: Optional[tuple[float, float]] = None,
    save_path: str | Path | None = None,
    publication_mode: bool = False,
) -> plt.Figure:
    """One line per variant, x = k, y = loss delta.  Needs ``by_k`` in *results*.
    
    Args:
        ylim: Optional y-axis limits as (ymin, ymax). Use to zoom into smaller deltas.
        publication_mode: If True, use publication-quality styling and save as PDF.
    """
    if publication_mode:
        set_publication_style()
        
    by_k = results.get("by_k", {})
    if not by_k:
        raise ValueError("No 'by_k' in results. Run with multiple --top-k values.")
    layer = results.get("config", {}).get("layer_idx", "?")
    k_independent = results.get("k_independent", {})

    k_vals = sorted(int(k) for k in by_k)
    
    # Merge k_independent variants into all k values for plotting
    merged_by_k = {}
    for k in k_vals:
        k_str = str(k)
        merged_by_k[k] = {**k_independent, **by_k.get(k_str, by_k.get(k, {}))}
    
    # Get all variants from merged data
    first = merged_by_k[k_vals[0]]
    all_variants = sorted(v for v in first if isinstance(first.get(v), dict) and "delta" in first[v])
    variants = variants or all_variants

    fig, ax = plt.subplots(figsize=figsize)
    acl = publication_mode
    colors = ACL_COLORS if acl else ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ACL_MARKERS if acl else ['o', 's', '^', 'D', 'v', 'p']
    lw = 1.8 if acl else 2.0
    ms = 5 if acl else 7
    mew = 0.8 if acl else 1.0

    for i, var in enumerate(variants):
        deltas = []
        for k in k_vals:
            entry = merged_by_k.get(k, {})
            d = entry.get(var, {}).get("delta")
            deltas.append(d)
        if all(d is not None for d in deltas):
            ax.plot(k_vals, deltas,
                    marker=markers[i % len(markers)],
                    label=var.replace('_', ' ').title(),
                    color=colors[i % len(colors)],
                    linewidth=lw,
                    markersize=ms,
                    markeredgewidth=mew,
                    markeredgecolor='white' if not acl else 'w',
                    zorder=2)
    ax.set_xlabel("Number of vectors $k$" if acl else "$k$ (number of singular vectors)")
    ax.set_ylabel("Loss increase $\\Delta$" if acl else "Loss $\\Delta$")
    if title:
        ax.set_title(title)
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6, zorder=0)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(
        frameon=True,
        loc='best',
        framealpha=0.95 if acl else 1.0,
        edgecolor='none' if acl else 'inherit',
        fancybox=False,
    )
    ax.grid(True, axis='y' if acl else 'both', alpha=0.35, linestyle='-', linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if acl:
        ax.xaxis.set_tick_params(width=0.8)
        ax.yaxis.set_tick_params(width=0.8)
    ax.set_xscale('log', base=2)
    ax.set_xticks(k_vals)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if len(k_vals) > 6:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        if publication_mode:
            pdf_path = save_path.with_suffix('.pdf') if save_path.suffix != '.pdf' else save_path
            fig.savefig(pdf_path, bbox_inches="tight", format='pdf')
            if pdf_path != save_path:
                print(f"Saved publication PDF: {pdf_path}")
            if save_path.suffix not in ('.pdf',):
                fig.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        
    return fig


# ---------------------------------------------------------------------------
# delta vs layer  (single k, multiple layers)
# ---------------------------------------------------------------------------

def plot_delta_vs_layers(
    results_list: List[Dict[str, Any]],
    k: int,
    *,
    variants: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 5),
    ylim: Optional[tuple[float, float]] = None,
    save_path: str | Path | None = None,
    publication_mode: bool = False,
) -> plt.Figure:
    """One line per variant, x = layer, y = loss delta for a fixed *k*.
    
    Args:
        ylim: Optional y-axis limits as (ymin, ymax). Use to zoom into smaller deltas.
        publication_mode: If True, use publication-quality styling and save as PDF.
    """
    if publication_mode:
        set_publication_style()
        
    by_layer: Dict[int, Dict[str, float]] = {}
    for res in results_list:
        layer = res.get("config", {}).get("layer_idx")
        if layer is None:
            continue
        
        # Merge k_independent and by_k[k] data
        k_independent = res.get("k_independent", {})
        by_k = res.get("by_k", {})
        k_entry = by_k.get(str(k), by_k.get(k, {}))
        
        # Combine k-independent and k-dependent data
        merged_entry = {**k_independent, **k_entry}
        if not merged_entry:
            continue
            
        by_layer[layer] = {v: merged_entry[v]["delta"] for v in merged_entry
                           if isinstance(merged_entry.get(v), dict) and "delta" in merged_entry[v]}
    if not by_layer:
        raise ValueError(f"No results with by_k[{k}]")

    layers = sorted(by_layer)
    all_vars = sorted({v for d in by_layer.values() for v in d})
    variants = variants or all_vars

    fig, ax = plt.subplots(figsize=figsize)
    acl = publication_mode
    colors = ACL_COLORS if acl else ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ACL_MARKERS if acl else ['o', 's', '^', 'D', 'v', 'p']
    lw = 1.8 if acl else 2.0
    ms = 5 if acl else 7
    mew = 0.8 if acl else 1.0

    for i, var in enumerate(variants):
        deltas = [by_layer[l].get(var) for l in layers]
        if any(d is None for d in deltas):
            continue
        ax.plot(layers, deltas,
                marker=markers[i % len(markers)],
                label=var.replace('_', ' ').title(),
                color=colors[i % len(colors)],
                linewidth=lw,
                markersize=ms,
                markeredgewidth=mew,
                markeredgecolor='white' if not acl else 'w',
                zorder=2)
    ax.set_xlabel("Layer" if acl else "Layer Index")
    ax.set_ylabel("Loss increase $\\Delta$" if acl else "Loss $\\Delta$")
    if title:
        ax.set_title(title)
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6, zorder=0)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(
        frameon=True,
        loc='best',
        framealpha=0.95 if acl else 1.0,
        edgecolor='none' if acl else 'inherit',
        fancybox=False,
    )
    ax.grid(True, axis='y' if acl else 'both', alpha=0.35, linestyle='-', linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if acl:
        ax.xaxis.set_tick_params(width=0.8)
        ax.yaxis.set_tick_params(width=0.8)
    ax.set_xticks(layers)
    fig.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        if publication_mode:
            pdf_path = save_path.with_suffix('.pdf') if save_path.suffix != '.pdf' else save_path
            fig.savefig(pdf_path, bbox_inches="tight", format='pdf')
            if pdf_path != save_path:
                print(f"Saved publication PDF: {pdf_path}")
            if save_path.suffix not in ('.pdf',):
                fig.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        
    return fig


# ---------------------------------------------------------------------------
# Ablation bar chart (single run)
# ---------------------------------------------------------------------------

def plot_ablation_results(
    results: Dict[str, Any],
    *,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Bar chart of loss + delta for each variant in a single-k run."""
    baseline = results.get("baseline_loss")
    by_k = results.get("by_k", {})
    # Pick first k
    if by_k:
        first_k = sorted(int(k) for k in by_k)[0]
        res = by_k.get(str(first_k), by_k.get(first_k, {}))
    else:
        res = results.get("results", {})
    if not res:
        raise ValueError("Empty results")

    variants = list(res.keys())
    losses = [res[v]["loss"] for v in variants]
    deltas = [res[v]["delta"] for v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    x = range(len(variants))
    palette = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]

    ax1.bar(x, losses, color=[palette[i % len(palette)] for i in x], edgecolor="k", lw=0.6)
    if baseline is not None:
        ax1.axhline(baseline, color="gray", ls="--", label="baseline")
        ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, rotation=45, ha="right")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss by variant")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, deltas, color=["#e74c3c" if d >= 0 else "#27ae60" for d in deltas], edgecolor="k", lw=0.6)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, rotation=45, ha="right")
    ax2.set_ylabel("Loss delta")
    ax2.set_title("Delta vs baseline")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Confusion heatmap (for vector interventions)
# ---------------------------------------------------------------------------

def plot_confusion_heatmap(
    matrix,
    *,
    token_ids: Optional[List[int]] = None,
    token_labels: Optional[List[str]] = None,
    figsize: tuple[float, float] = (8, 7),
    title: str = "Prediction before vs after",
    cmap: str = "Blues",
    save_path: str | Path | None = None,
) -> plt.Figure:
    mat = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mat, cmap=cmap, aspect="auto")
    k = mat.shape[0]
    labels = token_labels or ([str(t) for t in token_ids] if token_ids else [str(i) for i in range(k)])
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels[:k], rotation=45, ha="right")
    ax.set_yticklabels(labels[:k])
    ax.set_xlabel("Predicted after"); ax.set_ylabel("Predicted before")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_confusion_from_results(
    results: Dict[str, Any],
    variant_key: str,
    **kwargs,
) -> plt.Figure:
    """Plot confusion matrix stored in ``results["results"][variant_key]``."""
    entry = results.get("results", {}).get(variant_key, {})
    if "confusion_matrix" not in entry:
        raise ValueError(f"No confusion_matrix for {variant_key!r}")
    return plot_confusion_heatmap(
        entry["confusion_matrix"],
        token_ids=entry.get("confusion_token_ids"),
        title=f"Top-1 predictions: {variant_key.replace('_', ' ')}",
        **kwargs,
    )


def plot_expert_migration_heatmap(
    matrix,
    *,
    num_experts: int = 8,
    source_expert: Optional[int] = None,
    figsize: tuple[float, float] = (8, 7),
    title: Optional[str] = None,
    cmap: str = "Blues",
    save_path: Optional[str | Path] = None,
    as_percent: bool = False,
) -> plt.Figure:
    """Plot 8x8 expert migration matrix (rows = original expert, cols = new expert).

    If as_percent is True, each row is normalized to 100% and annotations show percentages.
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.shape != (num_experts, num_experts):
        mat = mat[:num_experts, :num_experts]
    if as_percent:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        mat = mat / row_sums * 100.0
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mat, cmap=cmap, aspect="auto")
    labels = [f"Exp {i}" for i in range(num_experts)]
    ax.set_xticks(range(num_experts))
    ax.set_yticks(range(num_experts))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("New expert")
    ax.set_ylabel("Original expert")
    if title is None and source_expert is not None:
        title = f"Migration matrix (source: Exp {source_expert})"
    if title:
        ax.set_title(title)
    for i in range(num_experts):
        for j in range(num_experts):
            val = mat[i, j]
            if val > 0:
                if as_percent:
                    text = f"{val:.1f}%"
                else:
                    text = str(int(val))
                ax.text(j, i, text, ha="center", va="center", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# ACL Paper Convenience Functions
# ---------------------------------------------------------------------------

def plot_confusion_heatmap_acl(
    matrix,
    *,
    column_width: str = "single",
    use_tex: bool = False,
    save_path: str | Path | None = None,
    **kwargs,
) -> plt.Figure:
    """ACL-formatted confusion heatmap (serif, column width, optional PDF)."""
    set_acl_style(use_tex=use_tex)
    if column_width == "single":
        figsize = (3.25, 3.0)
    elif column_width == "double":
        figsize = (6.75, 5.0)
    else:
        raise ValueError(f"column_width must be 'single' or 'double', got {column_width!r}")
    return plot_confusion_heatmap(
        matrix, figsize=figsize, save_path=save_path, cmap=_LIGHT_BLUES, **kwargs
    )


def plot_expert_migration_heatmap_acl(
    matrix,
    *,
    column_width: str = "single",
    use_tex: bool = False,
    save_path: str | Path | None = None,
    **kwargs,
) -> plt.Figure:
    """ACL-formatted expert migration heatmap (serif, column width, optional PDF)."""
    set_acl_style(use_tex=use_tex)
    if column_width == "single":
        figsize = (3.25, 3.0)
    elif column_width == "double":
        figsize = (6.75, 5.0)
    else:
        raise ValueError(f"column_width must be 'single' or 'double', got {column_width!r}")
    return plot_expert_migration_heatmap(
        matrix, figsize=figsize, save_path=save_path, cmap=_LIGHT_BLUES, **kwargs
    )


def plot_delta_vs_k_acl(
    results: Dict[str, Any],
    *,
    variants: Optional[List[str]] = None,
    column_width: str = "single",  # "single" or "double"
    ylim: Optional[tuple[float, float]] = None,
    save_path: str | Path | None = None,
    show_title: bool = False,
    use_tex: bool = False,
) -> plt.Figure:
    """
    Create ACL-formatted plot of loss delta vs k (PDF, column width, serif/LaTeX).
    
    Args:
        column_width: "single" (3.25") or "double" (6.75")
        show_title: If False (default), no title (use LaTeX caption)
        use_tex: If True, render text with LaTeX (matches paper font; requires pdflatex).
                 If False, use serif fonts and embed in PDF (vector, crisp).
    """
    set_acl_style(use_tex=use_tex)
    
    if column_width == "single":
        figsize = (3.25, 2.5)
    elif column_width == "double":
        figsize = (6.75, 4.0)
    else:
        raise ValueError(f"column_width must be 'single' or 'double', got {column_width!r}")
    
    return plot_delta_vs_k(
        results,
        variants=variants,
        title=None if not show_title else None,
        figsize=figsize,
        ylim=ylim,
        save_path=save_path,
        publication_mode=True,
    )


def plot_delta_vs_layers_acl(
    results_list: List[Dict[str, Any]],
    k: int,
    *,
    variants: Optional[List[str]] = None,
    column_width: str = "double",  # Usually double for layer plots
    ylim: Optional[tuple[float, float]] = None,
    save_path: str | Path | None = None,
    show_title: bool = False,
    use_tex: bool = False,
) -> plt.Figure:
    """
    Create ACL-formatted plot of loss delta vs layers (PDF, column width, serif/LaTeX).
    
    Args:
        column_width: "single" (3.25") or "double" (6.75")
        show_title: If False (default), no title (use LaTeX caption)
        use_tex: If True, render text with LaTeX (requires pdflatex). If False, serif + PDF.
    """
    set_acl_style(use_tex=use_tex)
    
    if column_width == "single":
        figsize = (3.25, 2.5)
    elif column_width == "double":
        figsize = (6.75, 4.0)
    else:
        raise ValueError(f"column_width must be 'single' or 'double', got {column_width!r}")
    
    return plot_delta_vs_layers(
        results_list,
        k,
        variants=variants,
        title=None if not show_title else None,
        figsize=figsize,
        ylim=ylim,
        save_path=save_path,
        publication_mode=True,
    )
