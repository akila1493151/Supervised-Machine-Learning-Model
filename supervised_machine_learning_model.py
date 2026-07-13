import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix, 
    classification_report, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score
)

# =========================================================================
# SETUP: Generate or Load Data
# =========================================================================
DATA_FILENAME = "cleaned_data.csv"

if not os.path.exists(DATA_FILENAME):
    print(f"--- Creating synthetic '{DATA_FILENAME}' for demonstration ---")
    # Generate mock continuous features and a target variable
    X_raw, y_raw = make_regression(n_samples=1200, n_features=8, noise=15.0, random_state=42)
    df = pd.DataFrame(X_raw, columns=[f"Feature_{i}" for i in range(8)])
    df['Target_Reg'] = y_raw
    
    # Introduce an ordinal categorical column
    # Order: Low (0), Medium (1), High (2)
    np.random.seed(42)
    df['Risk_Level'] = np.random.choice(['Low', 'Medium', 'High'], size=len(df), p=[0.3, 0.5, 0.2])
    
    # Introduce a nominal categorical column
    df['City_Group'] = np.random.choice(['Metro_A', 'Metro_B', 'Regional'], size=len(df))
    
    df.to_csv(DATA_FILENAME, index=False)

print(f"Loading dataset: {DATA_FILENAME}")
df_clean = pd.read_csv(DATA_FILENAME)

# =========================================================================
# TASK 1: Define Target Variables & Core Schema
# =========================================================================
print("\n=== Task 1: Defining Target Variables ===")
# Regression Target
y_reg = df_clean['Target_Reg']

# Classification Target (Binarized at the median)
median_val = y_reg.median()
y_clf = (y_reg > median_val).astype(int)

# Matrix X contains all columns except targets
X = df_clean.drop(columns=['Target_Reg'])

print(f"Dataset shape: {X.shape}")
print(f"Regression target distribution summary:\n{y_reg.describe()}")
print(f"Classification target class distribution:\n{y_clf.value_counts(normalize=True)}")

# =========================================================================
# TASK 2: Categorical Column Encoding
# =========================================================================
print("\n=== Task 2: Categorical Column Encoding ===")

# 1. Ordinal Encoding (Natural order: Low < Medium < High)
ordinal_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
X['Risk_Level'] = X['Risk_Level'].map(ordinal_mapping)
print("-> Applied Ordinal Mapping to 'Risk_Level' column.")

# 2. One-Hot Encoding (Nominal variable: City_Group)
# Dropping first dummy column to avoid multicollinearity
X = pd.get_dummies(X, columns=['City_Group'], drop_first=True, dtype=int)
print("-> Applied One-Hot Encoding to 'City_Group' (dropped first dummy column).")
print(f"Post-encoding feature matrix columns: {list(X.columns)}")

# =========================================================================
# TASK 3: Leak-Free Split & Feature Scaling
# =========================================================================
print("\n=== Task 3: Leak-Free Train-Test Split and Scaling ===")

# Split for Regression
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# Split for Classification (Using identical random state for alignment)
_, _, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# Fit scaler strictly on the training features matrix to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training shapes: X={X_train_scaled.shape}, y_reg={y_reg_train.shape}, y_clf={y_clf_train.shape}")
print(f"Testing shapes : X={X_test_scaled.shape}, y_reg={y_reg_test.shape}, y_clf={y_clf_test.shape}")

# =========================================================================
# TASK 4: Regression Modeling — Linear Regression & Ridge Comparison
# =========================================================================
print("\n=== Task 4: Regression Modeling ===")

# Plain OLS Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_reg_train)
y_pred_reg_lr = lr_model.predict(X_test_scaled)

mse_lr = mean_squared_error(y_reg_test, y_pred_reg_lr)
r2_lr = r2_score(y_reg_test, y_pred_reg_lr)

print("\n[Ordinary Least Squares (OLS) Linear Regression Results]")
print(f"Mean Squared Error (MSE): {mse_lr:.4f}")
print(f"R² Score                : {r2_lr:.4f}")

# Model Coefficients mapping
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
})
coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
top_3_features = coefficients.sort_values(by='Abs_Coefficient', ascending=False).head(3)

print("\nModel Coefficients:")
print(coefficients[['Feature', 'Coefficient']].to_string(index=False))
print("\nTop 3 Features with Largest Absolute Impact:")
print(top_3_features[['Feature', 'Coefficient']])

# Ridge Regularization
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_reg_train)
y_pred_reg_ridge = ridge_model.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_reg_test, y_pred_reg_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_reg_ridge)

print("\n[Ridge Regression Results]")
print(f"Mean Squared Error (MSE): {mse_ridge:.4f}")
print(f"R² Score                : {r2_ridge:.4f}")

# =========================================================================
# TASK 5: Classification Model — Logistic Regression Baseline
# =========================================================================
print("\n=== Task 5: Classification Modeling ===")

# Check class balance
minority_ratio = y_clf_train.value_counts(normalize=True).min()
print(f"Initial Minority Class Representation in Training Set: {minority_ratio * 100:.2f}%")

# Address class imbalance dynamically if the ratio drops below 35%
clf_kwargs = {'max_iter': 1000, 'random_state': 42}
if minority_ratio < 0.35:
    print("-> Minority class balance is below 35%. Activating class_weight='balanced'.")
    clf_kwargs['class_weight'] = 'balanced'
else:
    print("-> Data is relatively well balanced (>35%). Proceeding with regular weights.")

# Train Baseline Logistic Regression Model (C=1.0)
log_reg_baseline = LogisticRegression(C=1.0, **clf_kwargs)
log_reg_baseline.fit(X_train_scaled, y_clf_train)

y_pred_clf_base = log_reg_baseline.predict(X_test_scaled)
y_prob_clf_base = log_reg_baseline.predict_proba(X_test_scaled)[:, 1]

print("\n[Logistic Regression Evaluation Matrix (C=1.0)]")
print("Confusion Matrix:")
print(confusion_matrix(y_clf_test, y_pred_clf_base))
print("\nClassification Report:")
print(classification_report(y_clf_test, y_pred_clf_base))

# Compute and Plot ROC Curve
fpr_base, tpr_base, _ = roc_curve(y_clf_test, y_prob_clf_base)
auc_base = roc_auc_score(y_clf_test, y_prob_clf_base)

plt.figure(figsize=(8, 6))
plt.plot(fpr_base, tpr_base, color='darkorange', lw=2, label=f'Baseline C=1.0 (AUC = {auc_base:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
# Annotate AUC score directly onto plot graph
plt.text(0.6, 0.4, f"AUC Value: {auc_base:.4f}", bbox=dict(facecolor='white', alpha=0.8))
plt.draw()  # Prepares the rendering pipeline
print(f"ROC-AUC Performance Score: {auc_base:.4f}")

# =========================================================================
# TASK 5b: Decision-Threshold Sensitivity Analysis
# =========================================================================
print("\n=== Task 5b: Decision-Threshold Sensitivity ===")

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
threshold_records = []

for t in thresholds:
    # Evaluate boolean binary conditions over customized threshold breaks
    t_predictions = (y_prob_clf_base >= t).astype(int)
    
    p_score = precision_score(y_clf_test, t_predictions, zero_division=0)
    r_score = recall_score(y_clf_test, t_predictions, zero_division=0)
    f1 = f1_score(y_clf_test, t_predictions, zero_division=0)
    
    threshold_records.append({
        'Threshold': f"{t:.2f}",
        'Precision': f"{p_score:.4f}",
        'Recall': f"{r_score:.4f}",
        'F1': f"{f1:.4f}"
    })

print("\nDecision Threshold Profile Matrix:")
print(pd.DataFrame(threshold_records).to_string(index=False))

# =========================================================================
# TASK 6: Regularization Experiment (Strong Penalty C=0.01)
# =========================================================================
print("\n=== Task 6: Regularization Experiment (C=0.01) ===")

log_reg_regularized = LogisticRegression(C=0.01, **clf_kwargs)
log_reg_regularized.fit(X_train_scaled, y_clf_train)

y_pred_clf_reg = log_reg_regularized.predict(X_test_scaled)
y_prob_clf_reg = log_reg_regularized.predict_proba(X_test_scaled)[:, 1]

p_base = precision_score(y_clf_test, y_pred_clf_base)
r_base = recall_score(y_clf_test, y_pred_clf_base)

p_reg = precision_score(y_clf_test, y_pred_clf_reg)
r_reg = recall_score(y_clf_test, y_pred_clf_reg)
auc_reg = roc_auc_score(y_clf_test, y_prob_clf_reg)

# Plot Regularized Model Variant over current active window figure canvas
plt.plot(roc_curve(y_clf_test, y_prob_clf_reg)[0], roc_curve(y_clf_test, y_prob_clf_reg)[1], 
         color='forestgreen', lw=2, linestyle=':', label=f'Regularized C=0.01 (AUC = {auc_reg:.4f})')
plt.legend(loc="lower right")
plt.savefig('classification_roc_comparison.png', dpi=300)
print("Saved performance graph visual asset as 'classification_roc_comparison.png'")
plt.show()

# =========================================================================
# TASK 7: Bootstrap Confidence Interval for AUC Difference
# =========================================================================
print("\n=== Task 7: Bootstrap Confidence Intervals ===")

np.random.seed(42)
n_bootstraps = 500
auc_diff_records = []
y_clf_test_arr = np.array(y_clf_test)

for i in range(n_bootstraps):
    # Sample row indices with replacement matching test vector sizes
    boot_indices = np.random.choice(len(y_clf_test_arr), size=len(y_clf_test_arr), replace=True)
    
    # Isolate bootstrap splits targets and matching probabilities arrays
    y_boot = y_clf_test_arr[boot_indices]
    prob_base_boot = y_prob_clf_base[boot_indices]
    prob_reg_boot = y_prob_clf_reg[boot_indices]
    
    # Require balanced sampling states locally inside the loop to calculate metrics cleanly
    if len(np.unique(y_boot)) < 2:
        continue
        
    auc_b = roc_auc_score(y_boot, prob_base_boot)
    auc_r = roc_auc_score(y_boot, prob_reg_boot)
    
    auc_diff_records.append(auc_b - auc_r)

mean_diff = np.mean(auc_diff_records)
ci_lower = np.percentile(auc_diff_records, 2.5)
ci_upper = np.percentile(auc_diff_records, 97.5)

print("\n[Bootstrap Assessment Logs]")
print(f"Mean AUC Difference (C=1.0 minus C=0.01) : {mean_diff:.5f}")
print(f"95% Confidence Interval Span               : [{ci_lower:.5f}, {ci_upper:.5f}]")

if ci_lower > 0 or ci_upper < 0:
    print("Conclusion: The 95% Confidence Interval EXCLUDES zero. Performance variations are statistically significant.")
else:
    print("Conclusion: The 95% Confidence Interval INCLUDES zero. Performance variations may be due to random sampling variations.")

print("\n--- Pipeline Execution Completed Smoothly ---")
