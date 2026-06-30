"""
Wine Quality Prediction Analysis
=================================
Predicting wine quality from physicochemical properties using:
  - Random Forest Classifier
  - Stochastic Gradient Descent (SGD) Classifier
  - Support Vector Classifier (SVC)

Dataset: WineQT.csv (1143 red wine samples, 11 chemical features, quality score 3-8)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score
)
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
FIG_DIR = "figures"

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("/mnt/user-data/uploads/WineQT.csv")
df = df.drop(columns=["Id"])  # not a predictive feature, just a row identifier

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum().sum()} total")
print(f"\nQuality distribution:\n{df['quality'].value_counts().sort_index()}")

# ---------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS / VISUALIZATION
# ---------------------------------------------------------------------------

# --- 2a. Quality distribution ---
plt.figure(figsize=(7, 5))
sns.countplot(x="quality", data=df, hue="quality", palette="viridis", legend=False)
plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_quality_distribution.png")
plt.close()

# --- 2b. Correlation heatmap ---
plt.figure(figsize=(11, 9))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, square=True)
plt.title("Correlation Matrix of Chemical Properties & Quality")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_correlation_heatmap.png")
plt.close()

print("\nCorrelation of each feature with quality (sorted):")
print(corr["quality"].drop("quality").sort_values(ascending=False))

# --- 2c. Density & acidity vs quality (highlighted chemical properties) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.boxplot(x="quality", y="alcohol", data=df, ax=axes[0, 0], hue="quality",
            palette="viridis", legend=False)
axes[0, 0].set_title("Alcohol Content vs Quality")

sns.boxplot(x="quality", y="volatile acidity", data=df, ax=axes[0, 1],
            hue="quality", palette="viridis", legend=False)
axes[0, 1].set_title("Volatile Acidity vs Quality")

sns.boxplot(x="quality", y="density", data=df, ax=axes[1, 0], hue="quality",
            palette="viridis", legend=False)
axes[1, 0].set_title("Density vs Quality")

sns.boxplot(x="quality", y="citric acid", data=df, ax=axes[1, 1], hue="quality",
            palette="viridis", legend=False)
axes[1, 1].set_title("Citric Acid vs Quality")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_chemical_properties_vs_quality.png")
plt.close()

# --- 2d. Density vs fixed acidity scatter, colored by quality ---
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df["density"], df["fixed acidity"], c=df["quality"],
                       cmap="viridis", alpha=0.7, edgecolor="k", linewidth=0.3)
plt.colorbar(scatter, label="Quality")
plt.xlabel("Density")
plt.ylabel("Fixed Acidity")
plt.title("Density vs Fixed Acidity (colored by Quality)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_density_vs_acidity_scatter.png")
plt.close()

# --- 2e. Feature distributions ---
features = [c for c in df.columns if c != "quality"]
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
axes = axes.flatten()
for i, col in enumerate(features):
    sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(col)
for j in range(len(features), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_feature_distributions.png")
plt.close()

print(f"\nSaved 5 EDA figures to '{FIG_DIR}/'")

# ---------------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------------
# Quality is heavily imbalanced (scores 3,4,8 are rare). We frame this as a
# 6-class classification problem (raw quality 3-8), which is the harder,
# more informative version of the task. A "good/bad" binary collapse is
# noted as an easier alternative at the end.

X = df.drop(columns=["quality"])
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 70)
print("TRAIN/TEST SPLIT")
print("=" * 70)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------------------
results = {}

# --- Random Forest (tree-based, doesn't need scaling, but we use scaled
#     data for consistency across models) ---
rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)  # RF is scale-invariant; fit on raw features
rf_pred = rf.predict(X_test)
results["Random Forest"] = rf_pred

# --- SGD Classifier (linear model, sensitive to scale -> needs scaling) ---
sgd = SGDClassifier(loss="log_loss", max_iter=2000, random_state=42,
                     n_jobs=-1, early_stopping=True)
sgd.fit(X_train_scaled, y_train)
sgd_pred = sgd.predict(X_test_scaled)
results["SGD Classifier"] = sgd_pred

# --- SVC (distance-based, needs scaling) ---
svc = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svc.fit(X_train_scaled, y_train)
svc_pred = svc.predict(X_test_scaled)
results["SVC"] = svc_pred

# ---------------------------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON (6-class: quality 3-8)")
print("=" * 70)

summary_rows = []
for name, pred in results.items():
    acc = accuracy_score(y_test, pred)
    f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, pred, average="weighted", zero_division=0)
    summary_rows.append({"Model": name, "Accuracy": acc,
                          "F1 (macro)": f1_macro, "F1 (weighted)": f1_weighted})
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, zero_division=0))

summary_df = pd.DataFrame(summary_rows).sort_values("Accuracy", ascending=False)
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(summary_df.to_string(index=False))
summary_df.to_csv("model_comparison_summary.csv", index=False)

# --- Cross-validation (5-fold) for robustness check ---
print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION (accuracy)")
print("=" * 70)
cv_results = {}
cv_results["Random Forest"] = cross_val_score(rf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
X_scaled_full = scaler.fit_transform(X)
cv_results["SGD Classifier"] = cross_val_score(sgd, X_scaled_full, y, cv=5, scoring="accuracy", n_jobs=-1)
cv_results["SVC"] = cross_val_score(svc, X_scaled_full, y, cv=5, scoring="accuracy", n_jobs=-1)
for name, scores in cv_results.items():
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# ---------------------------------------------------------------------------
# 6. CONFUSION MATRICES
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
quality_labels = sorted(y.unique())
for ax, (name, pred) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, pred, labels=quality_labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=quality_labels, yticklabels=quality_labels, ax=ax)
    ax.set_title(f"{name}\nAccuracy: {accuracy_score(y_test, pred):.3f}")
    ax.set_xlabel("Predicted Quality")
    ax.set_ylabel("Actual Quality")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_confusion_matrices.png")
plt.close()

# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)
print(importances)

plt.figure(figsize=(9, 6))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            palette="viridis", legend=False)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_feature_importance.png")
plt.close()

# ---------------------------------------------------------------------------
# 8. MODEL ACCURACY COMPARISON BAR CHART
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.barplot(x="Model", y="Accuracy", data=summary_df, hue="Model",
            palette="muted", legend=False)
plt.title("Model Accuracy Comparison (6-class quality prediction)")
plt.ylim(0, 1)
for i, v in enumerate(summary_df["Accuracy"]):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_model_accuracy_comparison.png")
plt.close()

print(f"\nAll figures saved to '{FIG_DIR}/'. Summary CSV saved as 'model_comparison_summary.csv'.")
print("\nDone.")
