"""
Train a Random Forest surrogate model and estimate prediction uncertainty using ensemble variance.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "./data/mp_dft_data_clean.csv"
MODEL_PATH = "./models/rf_surrogate_uq.pkl"
RESULTS_PATH = "./results/rf_results_uq.txt"

# Load data
df = pd.read_csv(DATA_PATH)
features = ["nsites", "volume", "density", "density_atomic"]
X = df[features]
y = df["nsites"]  # Replace with target property if available

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ensemble of Random Forests for UQ
n_ensemble = 5
ensemble = []
all_preds = []
for seed in range(n_ensemble):
    model = RandomForestRegressor(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    ensemble.append(model)
    all_preds.append(model.predict(X_test))
all_preds = np.array(all_preds)

# Mean prediction and uncertainty (variance)
y_pred_mean = np.mean(all_preds, axis=0)
y_pred_std = np.std(all_preds, axis=0)
mae = mean_absolute_error(y_test, y_pred_mean)
r2 = r2_score(y_test, y_pred_mean)

# Save ensemble and results
joblib.dump(ensemble, MODEL_PATH)
with open(RESULTS_PATH, "w") as f:
    f.write(f"MAE: {mae}\nR2: {r2}\nMean prediction std (UQ): {np.mean(y_pred_std)}\n")


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Histogram of prediction errors
errors = np.abs(y_test - y_pred_mean)
plt.figure(figsize=(6, 4))
sns.histplot(errors, bins=30, kde=True)
plt.title('Histogram of Prediction Errors')
plt.xlabel('Absolute Error')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('./results/prediction_error_hist_uq.png')
plt.close()

# 2. True vs Predicted scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_mean, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted')
plt.tight_layout()
plt.savefig('./results/true_vs_pred_scatter_uq.png')
plt.close()

# 3. Uncertainty histogram
plt.figure(figsize=(6, 4))
sns.histplot(y_pred_std, bins=30, kde=True)
plt.title('Histogram of Prediction Uncertainty (std)')
plt.xlabel('Predicted Std')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('./results/prediction_uncertainty_hist_uq.png')
plt.close()

print(f"Ensemble model saved to {MODEL_PATH}")
print(f"Results saved to {RESULTS_PATH}")
print(f"MAE: {mae}")
print(f"R2: {r2}")
print(f"Mean prediction std (UQ): {np.mean(y_pred_std)}")
print("Saved visualizations to ./results/")
