import os
import sys
import pandas as pd

# Ensure parent is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compare_results():
    """
    Saves the comparative metrics from results.csv into comparison.xlsx
    and generates visual graphs under results/figures/ for all key metrics.
    Also produces a structured results/error_analysis.csv report.
    """
    csv_path = "results/results.csv"
    xlsx_path = "results/comparison.xlsx"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run evaluation/evaluate.py first.", file=sys.stderr)
        return
        
    # 1. Read comparative metrics
    df = pd.read_csv(csv_path)
    # results/comparison.xlsx is already generated with 9 sheets by evaluation/evaluate.py.
    print(f"Using metrics from results.csv for plotting.")
    
    # 2. Generate Figures using Matplotlib
    try:
        import matplotlib
        # Configure Agg backend to render non-interactively
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        os.makedirs("results/figures", exist_ok=True)
        
        metrics = ["BLEU", "chrF++", "Semantic Similarity", "IKS Preservation", "Context Preservation"]
        sys_col = "System" if "System" in df.columns else "Model"
        systems = df[sys_col].tolist()
        
        for metric in metrics:
            if metric not in df.columns:
                continue
                
            values = df[metric].tolist()
            clean_values = []
            for val in values:
                try:
                    clean_values.append(float(val))
                except ValueError:
                    clean_values.append(0.0)
                    
            plt.figure(figsize=(8, 5))
            colors = ["#3498db", "#2ecc71", "#9b59b6"]
            bars = plt.bar(systems, clean_values, color=colors[:len(systems)], width=0.45)
            
            # Styling
            plt.title(f"{metric} Score Comparison", fontsize=12, fontweight="bold", pad=15)
            plt.ylabel("Score (%)", fontsize=10)
            plt.ylim(0, 110)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Value markers on top of bars
            for bar, orig_val in zip(bars, values):
                height = bar.get_height()
                label = f"{height:.2f}%" if height > 0 else str(orig_val)
                plt.text(bar.get_x() + bar.get_width()/2.0, height + 2, label, ha='center', va='bottom', fontsize=9, fontweight='bold')
                
            plt.tight_layout()
            chart_path = f"results/figures/{metric.lower().replace(' ', '_').replace('+', '')}_comparison.png"
            plt.savefig(chart_path, dpi=300)
            plt.close()
            print(f"Generated metric figure: {chart_path}")
            
    except ImportError:
        print("Warning: matplotlib not installed. Bypassing chart image generation.", file=sys.stderr)
        
    # 3. Error Analysis Report
    print("Error analysis is already managed by evaluation/evaluate.py")

if __name__ == "__main__":
    compare_results()
