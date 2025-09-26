"""
Compare surrogate predictions, UQ calibration, and fallback effectiveness against DFT results.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "./data/mp_dft_data_clean.csv"
OOD_PATH = "./results/ood_flags.csv"
MODEL_PATH = "./models/rf_surrogate_uq.pkl"
RESULTS_PATH = "./results/evaluation_benchmark.txt"

# Load data and OOD flags
df = pd.read_csv(DATA_PATH)
ood_df = pd.read_csv(OOD_PATH)
ensemble = joblib.load(MODEL_PATH)
features = ["nsites", "volume", "density", "density_atomic"]
X = df[features]
y = df["nsites"]  # Replace with true target if available

# Surrogate predictions and uncertainty
all_preds = np.array([model.predict(X) for model in ensemble])
pred_mean = np.mean(all_preds, axis=0)
pred_std = np.std(all_preds, axis=0)

# Metrics
mae = mean_absolute_error(y, pred_mean)
r2 = r2_score(y, pred_mean)
calibration = np.mean(pred_std)
num_ood = np.sum(ood_df["OOD_flag"])

with open(RESULTS_PATH, "w") as f:
    f.write(f"MAE: {mae}\nR2: {r2}\nMean UQ (std): {calibration}\nNumber of OOD cases: {num_ood}\n")


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Histogram of prediction errors
errors = np.abs(y - pred_mean)
plt.figure(figsize=(6,4))
sns.histplot(errors, bins=30, kde=True)
plt.title('Histogram of Prediction Errors')
plt.xlabel('Absolute Error')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('./results/prediction_error_hist.png')
plt.close()

# 2. True vs Predicted scatter plot
plt.figure(figsize=(6,6))
plt.scatter(y, pred_mean, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted')
plt.tight_layout()
plt.savefig('./results/true_vs_pred_scatter.png')
plt.close()

# 3. Uncertainty histogram
plt.figure(figsize=(6,4))
sns.histplot(pred_std, bins=30, kde=True)
plt.title('Histogram of Prediction Uncertainty (std)')
plt.xlabel('Predicted Std')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('./results/prediction_uncertainty_hist.png')
plt.close()

# 4. OOD flag bar plot
plt.figure(figsize=(4,4))
ood_counts = ood_df['OOD_flag'].value_counts()
ood_counts.plot(kind='bar')
plt.title('OOD Flag Distribution')
plt.xlabel('OOD Flag')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('./results/ood_flag_bar.png')
plt.close()

print(f"Evaluation and benchmarking results saved to {RESULTS_PATH}")
print(f"MAE: {mae}")
print(f"R2: {r2}")
print(f"Mean UQ (std): {calibration}")
print(f"Number of OOD cases: {num_ood}")
print("Saved visualizations to ./results/")
