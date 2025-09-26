"""
Detect out-of-distribution (OOD) predictions using ensemble uncertainty and trigger fallback.
"""
import pandas as pd
import numpy as np
import joblib

DATA_PATH = "./data/mp_dft_data_clean.csv"
MODEL_PATH = "./models/rf_surrogate_uq.pkl"
OOD_PATH = "./results/ood_flags.csv"

# Load data and model
X = pd.read_csv(DATA_PATH)[["nsites", "volume", "density", "density_atomic"]]
ensemble = joblib.load(MODEL_PATH)

# Predict and compute uncertainty
all_preds = np.array([model.predict(X) for model in ensemble])
pred_mean = np.mean(all_preds, axis=0)
pred_std = np.std(all_preds, axis=0)

# Flag OOD if uncertainty exceeds threshold
threshold = np.percentile(pred_std, 95)  # Top 5% uncertainty
ood_flags = pred_std > threshold

# Save OOD flags and uncertainty
ood_df = pd.DataFrame({
    "pred_mean": pred_mean,
    "pred_std": pred_std,
    "OOD_flag": ood_flags
})
ood_df.to_csv(OOD_PATH, index=False)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Histogram of prediction uncertainty
plt.figure(figsize=(6,4))
sns.histplot(pred_std, bins=30, kde=True)
plt.title('Histogram of Prediction Uncertainty (std)')
plt.xlabel('Predicted Std')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('./results/uncertainty_hist.png')
plt.close()

# 2. OOD flag bar plot
plt.figure(figsize=(4,4))
pd.Series(ood_flags).value_counts().plot(kind='bar')
plt.title('OOD Flag Distribution')
plt.xlabel('OOD Flag')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('./results/ood_flag_bar.png')
plt.close()

print(f"OOD flags and uncertainty saved to {OOD_PATH}")
print(f"Number of OOD cases: {np.sum(ood_flags)}")
print("Saved visualizations to ./results/")
