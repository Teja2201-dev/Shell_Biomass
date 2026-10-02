import pandas as pd
import matplotlib.pyplot as plt

INPUT = "outputs/ablation_study_results.csv"

df = pd.read_csv(INPUT)

# Model performance comparison
plt.figure(figsize=(9, 6))

plt.bar(
    df["Feature Set"],
    df["R2"]
)

plt.ylabel("R² Score")
plt.xlabel("Feature Set")
plt.title("Ablation Study: R² Comparison")
plt.xticks(rotation=20, ha="right")

plt.tight_layout()

plt.savefig(
    "outputs/ablation_r2_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("FINAL PLOT CREATED")
print("Saved: outputs/ablation_r2_comparison.png")