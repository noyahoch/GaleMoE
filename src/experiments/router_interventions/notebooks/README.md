# Creating Publication-Quality Plots for ACL Papers

## ACL Format Requirements

### Figure Dimensions
- **Single column**: 3.25 inches wide (8.26 cm)
- **Double column**: 6.75 inches wide (17.15 cm)
- **Height**: Typically 2-4 inches, maintaining good aspect ratio

### Font Guidelines
- **Main text**: 10pt (ACL papers use 10pt body text)
- **Axis labels**: 9-10pt
- **Tick labels**: 8-9pt
- **Legend**: 8-9pt
- **Title**: Can be omitted (goes in LaTeX caption instead)

### File Format
- **Preferred**: PDF (vector graphics, scales perfectly)
- **Alternative**: PNG at 300+ DPI
- **Avoid**: JPEG (lossy compression, poor for line plots)

## Quick Start

```python
from src.experiments.router_interventions.viz.plotter import (
    plot_delta_vs_k, 
    plot_delta_vs_layers,
    set_publication_style
)

# Method 1: Use publication_mode flag
fig = plot_delta_vs_k(
    results, 
    publication_mode=True,
    figsize=(3.25, 2.5),  # Single column width
    save_path="figures/loss_delta_vs_k.png"
)

# Method 2: Set style globally
set_publication_style()
fig = plot_delta_vs_k(results, figsize=(3.25, 2.5))
fig.savefig("figures/loss_delta_vs_k.pdf", bbox_inches='tight', dpi=300)
```

## LaTeX Integration

### In your ACL paper (.tex):

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/loss_delta_vs_k.pdf}
\caption{Loss $\Delta$ as a function of the number of singular vectors $k$ 
         projected out from router weights (Layer 20). The SVD variant shows 
         the strongest effect, with loss delta increasing monotonically with $k$.}
\label{fig:loss-delta-k}
\end{figure}
```

### For double-column figures:

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/loss_delta_vs_layers.pdf}
\caption{Loss $\Delta$ across all 32 layers of Mixtral-8x7B for different 
         intervention variants with $k=128$ singular vectors.}
\label{fig:loss-delta-layers}
\end{figure*}
```

## Best Practices

### 1. Remove Titles
Don't include titles in the figure itself - use LaTeX captions:
```python
fig = plot_delta_vs_k(results, title=None)  # or title=""
```

### 2. Use Appropriate Sizes
```python
# Single column (most common)
figsize=(3.25, 2.5)

# Double column (for complex multi-panel figures)
figsize=(6.75, 4.0)

# Aspect ratios
# - 4:3 ratio: (3.25, 2.44) for single column
# - 16:9 ratio: (3.25, 1.83) for single column
# - Golden ratio: (3.25, 2.01) for single column
```

### 3. Simplify Legends
```python
# Rename variants for clarity
variants = ['SVD', 'Orthogonal', 'Random', 'Zero', 'Shuffle']
fig = plot_delta_vs_k(results, variants=variants)
```

### 4. Export Both PNG and PDF
```python
# PDF for paper (vector, scales perfectly)
fig.savefig("figures/loss_delta_vs_k.pdf", bbox_inches='tight', dpi=300)

# PNG for presentations/websites
fig.savefig("figures/loss_delta_vs_k.png", bbox_inches='tight', dpi=300)
```

## Example: Complete Workflow

```python
import matplotlib.pyplot as plt
from src.experiments.router_interventions.viz.plotter import *

# Load results
results = load_results("results/project_out_L20_k1-2-4-8-16-32-64-128-256-512.json")

# Set publication style globally
set_publication_style()

# Create figure (single column width)
fig = plot_delta_vs_k(
    results,
    variants=['svd', 'random', 'shuffle', 'zero'],
    title=None,  # No title - will use LaTeX caption
    figsize=(3.25, 2.5),
    ylim=(0, 0.035),  # Zoom into interesting range
)

# Save both formats
fig.savefig("figures/loss_delta_vs_k.pdf", bbox_inches='tight', dpi=300, format='pdf')
fig.savefig("figures/loss_delta_vs_k.png", bbox_inches='tight', dpi=300)

plt.close(fig)
```

## Color Recommendations

The current color palette is colorblind-friendly and works well in grayscale:
- Blue (#1f77b4)
- Orange (#ff7f0e)
- Green (#2ca02c)
- Red (#d62728)
- Purple (#9467bd)
- Brown (#8c564b)

For 2-3 variants, consider using distinct line styles instead of just colors:
- Solid line (-)
- Dashed line (--)
- Dash-dot line (-.)

## Troubleshooting

### Text too small in final paper?
Increase figure size or font sizes:
```python
plt.rcParams.update({'font.size': 11})
```

### Legend overlapping data?
```python
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
```

### Figure too wide for single column?
Reduce width to exactly 3.25 inches:
```python
figsize=(3.25, 2.5)
```

### Math symbols not rendering?
Use LaTeX notation:
```python
ax.set_ylabel(r'Loss $\Delta$')
ax.set_xlabel(r'$k$ (singular vectors)')
```
