"""
Train ML surrogate models for DFT energies/forces prediction using the preprocessed dataset.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "./data/mp_dft_data_clean.csv"
MODEL_PATH = "./models/rf_surrogate.pkl"
RESULTS_PATH = "./results/rf_results.txt"

# Load data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Example feature engineering: use nsites, volume, density, density_atomic as features
features = ["nsites", "volume", "density", "density_atomic"]
X = df[features]
y = df["nsites"]  # Replace with target property (e.g., energy) if available

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save model and results
joblib.dump(model, MODEL_PATH)
with open(RESULTS_PATH, "w") as f:
    f.write(f"MAE: {mae}\nR2: {r2}\n")


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Histogram of prediction errors
errors = abs(y_test - y_pred)
plt.figure(figsize=(6, 4))
sns.histplot(errors, bins=30, kde=True)
plt.title('Histogram of Prediction Errors')
plt.xlabel('Absolute Error')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('./results/prediction_error_hist.png')
plt.close()

# 2. True vs Predicted scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted')
plt.tight_layout()
plt.savefig('./results/true_vs_pred_scatter.png')
plt.close()

print(f"Model saved to {MODEL_PATH}")
print(f"Results saved to {RESULTS_PATH}")
print(f"MAE: {mae}")
print(f"R2: {r2}")
print("Saved visualizations to ./results/")
