"""Main entry point for router-expert alignment analysis."""
import pandas as pd

from src.analysis import (
    AlignmentRunner,
    SVDAlignmentAnalyzer,
    save_results,
)
from src.models import Mixtral8x7B

if __name__ == "__main__":
    # Initialize model and analyzer
    moe = Mixtral8x7B()
    # For Mixtral8x7B, can use up to 4096 (intermediate_size=14336, rank=4096)
    analyzer = SVDAlignmentAnalyzer(
        k_list=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
        n_shuffles=200,
        seed=42,
    )
    runner = AlignmentRunner(moe=moe, analyzer=analyzer)

    # Run analysis
    for layer in range(moe.n_layers):   
        print(f"Running analysis on Layer {layer}")
        results = runner.run_layer(layer=layer)

        # Save results with metadata
        saved_path = save_results(results, analyzer, layer, moe.model_id)
        
        # Display summary
        pd.set_option('display.max_columns', 20)
        df = pd.DataFrame([r.__dict__ for r in results])
        print("\nSummary:")
        
        # Standard metrics for all k values (don't include cos_squared in summary table)
        summary_cols = ["align", "delta_vs_shuffle", "z_vs_shuffle", "effect_over_random"]
        print(df.groupby("k")[summary_cols].mean())
        
        print(f"\nTo compare with other runs, use:")
        print(f"  from src.analysis import compare_runs")
        print(f"  compare_runs(['results/file1.json', 'results/file2.json', ...])")
        
        # Print cos_squared values at the end (only once, comprehensive)
        if 1 in df["k"].values:
            k1_results = df[df["k"] == 1]
            if len(k1_results) > 0:
                print("\n" + "=" * 60)
                print("Cos²(θ) Alignment Summary (k=1):")
                print("=" * 60)
                print(f"Mean cos²(θ): {k1_results['cos_squared'].mean():.6f}")
                print(f"Max cos²(θ):  {k1_results['cos_squared'].max():.6f}")
                print(f"Min cos²(θ):  {k1_results['cos_squared'].min():.6f}")
                print(f"Std cos²(θ):  {k1_results['cos_squared'].std():.6f}")
                print("\nPer-expert cos²(θ) values:")
                for _, row in k1_results.iterrows():
                    print(f"  Expert {int(row['expert'])}: {row['cos_squared']:.6f}")
                print("=" * 60)