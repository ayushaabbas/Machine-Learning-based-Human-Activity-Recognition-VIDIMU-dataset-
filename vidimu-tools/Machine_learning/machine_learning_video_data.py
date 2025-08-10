import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier  # ok if not installed; you can remove if needed

from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning
import warnings

# =========================
# Helper: simple time-series features
# =========================
def zero_crossing_rate(data: np.ndarray) -> int:
    """
    How often the signal crosses zero. Treat 0 as negative to avoid ambiguity.
    """
    signs = np.sign(data)
    signs[signs == 0] = -1
    return int(np.sum(np.diff(signs) != 0))

def mean_crossing_rate(data: np.ndarray) -> int:
    """
    How often the signal crosses its mean value.
    """
    mean = float(np.mean(data))
    signs = np.sign(data - mean)
    return int(np.sum(np.diff(signs) != 0))

# =========================
# STEP 1: Load and select files
# =========================
# Your dataset root (VIDEO CSVs per subject)
data_folder = r"D:\Machine Learning\Video_IMU data\Data\dataset\videoandimus"

# Map (subject, activity) -> exactly one CSV (prefer T01)
file_map = defaultdict(dict)

# We expect video CSVs named like: S01_A01_T01.csv (activity code at index 1, trial at index 2)
for subject in os.listdir(data_folder):
    subject_path = os.path.join(data_folder, subject)
    if os.path.isdir(subject_path):
        for file in os.listdir(subject_path):
            # Only CSVs; skip BodyTrack "Npose" if present
            if file.lower().endswith(".csv") and "npose" not in file.lower():
                parts = file.split("_")
                # Expect something like: ["S01", "A01", "T01.csv"]
                if len(parts) >= 3:
                    activity = parts[1]                 # A01, A02, ...
                    trial = parts[2].split(".")[0]      # T01, T02, ...
                    key = (subject, activity)
                    # Prefer T01 for each (subject, activity). If T01 not seen, store first we see.
                    if trial == "T01" or key not in file_map:
                        file_map[key] = os.path.join(subject_path, file)

csv_files = list(file_map.values())
subjects = [os.path.basename(os.path.dirname(f)) for f in csv_files]
print(f"Found {len(csv_files)} unique subject-activity pairs with one trial each.")

# =========================
# STEP 2: Feature aggregation (build X, y, groups)
# =========================
all_data = []     # rows of features
labels = []       # activity labels (merged)
subject_ids = []  # subject IDs (for LOSO groups)

# Merge codes → human-readable labels you used before
activity_merges = {
    'A01': 'walk',
    'A02': 'walk',
    'A03': 'walk_line',
    'A04': 'sit_stand',
    'A05': 'move_bottle',
    'A06': 'move_bottle',
    'A07': 'drink',
    'A08': 'drink',
    'A09': 'lego',
    'A10': 'throw',
    'A11': 'reach',
    'A12': 'reach',
    'A13': 'tear_throw'
}

# We’ll build readable feature names once from the **first** file we manage to process,
# using the numeric columns found in that file (and the stat list below).
feature_names = None
STAT_LIST = ['mean', 'std', 'min', 'max', 'var', 'median', 'zcr', 'mcr']

for file, subj in zip(csv_files, subjects):
    try:
        # Read CSV
        df = pd.read_csv(file)
        # Drop metadata-ish columns if present
        df = df.drop(columns=[col for col in df.columns
                              if 'Unnamed' in col or 'Time' in col or 'ID' in col],
                     errors='ignore')

        # Derive activity label
        activity_code = os.path.basename(file).split("_")[1]  # assumes S01_A01_T01.csv
        label = activity_merges.get(activity_code, activity_code)

        # Only numeric columns are used for stats
        num_cols = df.select_dtypes(include=np.number).columns

        # Build feature names from the **first** processed file,
        # so later we can show proper names in feature importance plots.
        # (If different files have different numeric columns, this keeps your original
        # logic but names will reflect the first file’s columns.)
        if feature_names is None:
            feature_names = []
            for col in num_cols:
                feature_names.extend([f"{stat}_{col}" for stat in STAT_LIST])

        # Compute stats in STAT_LIST order for each numeric column
        stats = []
        for col in num_cols:
            x = df[col].dropna().values.astype(float)
            stats.extend([
                np.mean(x), np.std(x), np.min(x), np.max(x),
                np.var(x), np.median(x), zero_crossing_rate(x), mean_crossing_rate(x)
            ])

        # If some file has a different number of numeric columns, fall back gracefully:
        # - If length matches the feature_names we built, use them.
        # - Otherwise, just create generic names for THIS row (keeps concat working).
        if len(stats) == len(feature_names):
            row_df = pd.DataFrame([stats], columns=feature_names)
        else:
            # Generic names if lengths mismatch
            row_df = pd.DataFrame([stats])

        all_data.append(row_df)
        labels.append(label)
        subject_ids.append(subj)

    except Exception as e:
        print(f"Failed to process {file}: {e}")

# Concatenate all 1-row DataFrames into a single table
X = pd.concat(all_data, ignore_index=True)

# If for some reason we never set feature_names (e.g., all files failed),
# or some rows had generic columns, set fallback names so downstream code is safe.
if feature_names is None or (X.columns.dtype != object):
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X.columns = feature_names

# Labels/Groups arrays
y = np.array(labels)
groups = np.array(subject_ids)

# Impute missing values with column means (simple and robust)
X.fillna(X.mean(), inplace=True)

# =========================
# STEP 3: Encode labels (A01→walk etc.)
# =========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Sample activity labels (merged):", labels[:5])
print("Encoded activity labels:", y_encoded[:5])
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# =========================
# PCA (visualisation only)
# =========================
print("\nRunning PCA Visualisation...")
scaler_all = StandardScaler()
X_scaled_all = scaler_all.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_all)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set2', legend='full', s=60)
plt.title("PCA Projection of Activities")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# =========================
# GroupShuffleSplit (sanity optional)
# =========================
print("\nPerforming a single GroupShuffleSplit for reference...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# =========================
# STEP 4: Models to compare
# =========================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=5000, solver='saga'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# =========================
# STEP 5: LOSO CV + plots
# =========================
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logo = LeaveOneGroupOut()
accuracies = []

for name, model in models.items():
    fold_accuracies = []
    fold_f1s = []
    all_y_true = []
    all_y_pred = []

    print(f"\n=== {name} ===")

    # Subject-wise folds
    for tr_idx, te_idx in logo.split(X, y_encoded, groups):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y_encoded[tr_idx], y_encoded[te_idx]

        # Scale inside each fold (fit on train only!)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Train & predict
        model.fit(X_tr_s, y_tr)
        y_hat = model.predict(X_te_s)

        # Metrics per fold
        fold_accuracies.append(accuracy_score(y_te, y_hat))
        fold_f1s.append(f1_score(y_te, y_hat, average='macro'))

        # For aggregated report/CM across folds
        all_y_true.extend(y_te)
        all_y_pred.extend(y_hat)

    mean_acc = float(np.mean(fold_accuracies))
    mean_f1 = float(np.mean(fold_f1s))

    print(f"Mean LOSO Accuracy: {mean_acc:.3f}")
    print(f"Mean Macro F1-score: {mean_f1:.3f}")
    accuracies.append((name, mean_acc))

    # ---- Text report
    print("\nAggregate Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_))

    # ---- Confusion matrix heatmap
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Aggregate Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ---- NEW: Feature importance bar plots (tree-based models only)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # pick top-k
        k = min(10, len(importances))
        top_idx = np.argsort(importances)[-k:]  # ascending order
        top_feats = [feature_names[i] for i in top_idx]
        top_vals = importances[top_idx]

        # Print nicely
        print("\nTop features:")
        for f, v in sorted(zip(top_feats, top_vals), key=lambda t: t[1], reverse=True):
            print(f"  {f}: {v:.4f}")

        # Plot bar chart
        plt.figure(figsize=(8, 5))
        sns.barplot(x=top_vals, y=top_feats, orient="h")
        plt.title(f"Top {k} Feature Importances — {name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

# =========================
# STEP 6: Compare models (bar)
# =========================
model_names = [item[0] for item in accuracies]
model_accuracies = [item[1] for item in accuracies]

plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=model_accuracies, hue=model_names, palette="Set2", legend=False)
plt.ylim(0, 1)
plt.title("Classifier Accuracies (LOSO Cross-Validation)")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.tight_layout()
plt.show()
