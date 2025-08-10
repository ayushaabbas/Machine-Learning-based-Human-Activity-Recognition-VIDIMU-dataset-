# === Human Activity Recognition on VIDIMU Dataset Using Random Forest and LOSO Cross-Validation ===
# === Import required libraries ===
import os                           # For navigating directories and handling file paths
import pandas as pd                 # For reading and manipulating tabular data (.csv, .tsv)
import numpy as np                  # For numerical operations, arrays, and basic math
import matplotlib.pyplot as plt     # For creating static visual plots (scatter, bar, heatmap, etc.)
import seaborn as sns               # For aesthetic, high-level plotting (heatmaps, clustered scatter, etc.)

from collections import defaultdict                     # For creating nested dictionaries with default values
from sklearn.ensemble import RandomForestClassifier     # Random Forest classifier for multi-class activity prediction
from sklearn.decomposition import PCA                   # Principal Component Analysis for dimensionality reduction
from sklearn.model_selection import LeaveOneGroupOut    # LOSO cross-validation based on subject IDs
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For scaling features and encoding labels
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score  # Evaluation metrics
from sklearn.exceptions import ConvergenceWarning       # Warning raised if model fails to converge
import warnings                                          # Suppress specific warning messages

# === Custom Feature Functions ===
def zero_crossing_rate(data):
    signs = np.sign(data)           # Convert data to -1, 0, or 1 depending on sign
    signs[signs == 0] = -1          # Treat zero as negative to avoid ambiguity
    return np.sum(np.diff(signs) != 0)  # Count number of sign changes (i.e., zero crossings)

def mean_crossing_rate(data):
    mean = np.mean(data)                 # Compute mean of the signal
    signs = np.sign(data - mean)         # Check whether each point is above or below the mean
    return np.sum(np.diff(signs) != 0)   # Count number of mean crossings

# === STEP 1: Load and select one .mot file per subject-activity ===
data_folder = r"D:\Machine Learning\Video_IMU data\Data\dataset\videoandimus"
file_map = {}
preferred_trials = ["T01", "T02", "T03", "T04", "T05"]  # Prioritised trials if multiple available

# Loop through each subject directory in the dataset
for subject in os.listdir(data_folder):
    subject_path = os.path.join(data_folder, subject)  # Full path to subject folder

    if os.path.isdir(subject_path):  # Check if path is a folder
        # List all .mot files in the folder
        files = [f for f in os.listdir(subject_path) if f.lower().endswith(".mot")]

        activity_trials = defaultdict(dict)  # Dict to map activity → trial → filepath

        for f in files:
            parts = f.split("_")  # e.g. S01_IMU_A01_T02.mot → ['S01', 'IMU', 'A01', 'T02.mot']
            if len(parts) >= 4:
                activity = parts[2]  # Extract activity code (A01, A02, ...)
                trial = parts[3].split(".")[0]  # Extract trial number (T01, T02, ...)
                activity_trials[activity][trial] = os.path.join(subject_path, f)

        # Choose one preferred trial per activity (T01 to T05)
        for activity, trials in activity_trials.items():
            for trial in preferred_trials:
                if trial in trials:
                    file_map[(subject, activity)] = trials[trial]
                    break  # Only one trial per activity is selected

print(f"Loaded {len(file_map)} subject-activity pairs from {len(set(k[0] for k in file_map))} subjects.")

# === STEP 2: Feature extraction ===
activity_merges = {
    'A01': 'walk', 'A02': 'walk', 'A03': 'walk_line',
    'A04': 'sit_stand', 'A05': 'move_bottle', 'A06': 'move_bottle',
    'A07': 'drink', 'A08': 'drink', 'A09': 'lego',
    'A10': 'throw', 'A11': 'reach', 'A12': 'reach', 'A13': 'tear_throw'
}

all_data, labels, subject_ids = [], [], []
feature_names = []  # Will be populated once using the first valid .mot file
example_feature = ['mean', 'std', 'min', 'max', 'var', 'median', 'zcr', 'mcr']

# Extract statistical features for each joint signal from each subject-activity file
for (subject, activity), file in file_map.items():
    try:
        df = pd.read_csv(file, sep='\t', skiprows=6)  # Load .mot file, skipping OpenSim headers
        df = df.drop(columns=[col for col in df.columns if 'time' in col], errors='ignore')
        activity_label = activity_merges.get(activity, activity)  # Convert A01 → walk, etc.

        stats = []
        columns = df.select_dtypes(include=np.number).columns  # Only process numerical columns

        # Generate feature names once, using the first valid file
        if not feature_names:
            for col in columns:
                feature_names.extend([f"{stat}_{col}" for stat in example_feature])

        for col in columns:
            x = df[col].dropna().values.astype(float)
            stats.extend([
                np.mean(x), np.std(x), np.min(x), np.max(x),
                np.var(x), np.median(x), zero_crossing_rate(x), mean_crossing_rate(x)
            ])

        all_data.append(pd.DataFrame([stats]))  # Store the feature vector
        labels.append(activity_label)
        subject_ids.append(subject)

    except Exception as e:
        print(f" Failed to process {file}: {e}")  # Handle corrupt or unreadable files

# Combine all feature vectors into a single DataFrame
X = pd.concat(all_data, ignore_index=True)
X.columns = feature_names  # Assign column names

# Convert labels and subject IDs to NumPy arrays for ML use
y = np.array(labels)
groups = np.array(subject_ids)

# Replace missing values with column-wise means
X.fillna(X.mean(), inplace=True)

# === STEP 3: Encode merged labels ===
le = LabelEncoder()                   # Convert string labels to integers
y_encoded = le.fit_transform(y)       # Encode labels (e.g., 'walk' → 0)

# Show examples of original and encoded labels
print("\nSample labels:", labels[:5])
print("Encoded labels:", y_encoded[:5])
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print("Subjects used:", sorted(set(subject_ids)))

# === PCA Cluster Plot ===
print("\n Running PCA Visualisation...")

# Standardise the features before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce features to 2D for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot PCA results to visualise activity clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set2', legend='full', s=60)
plt.title("PCA Projection of Activities")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# === Random Forest with LOSO Cross-Validation ===
if len(set(subject_ids)) < 2:
    print("ERROR: At least 2 unique subjects required for LOSO.")
    exit()

# Suppress convergence warnings from scikit-learn
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Prepare LOSO cross-validation and tracking lists
logo = LeaveOneGroupOut()
fold_accuracies = []
fold_f1s = []
all_y_true = []
all_y_pred = []

model = RandomForestClassifier(n_estimators=100, random_state=42)
print("\n Running Random Forest with LOSO...")

# Perform one fold per subject
for train_idx, test_idx in logo.split(X, y_encoded, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # Standardise train and test data separately (fit only on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)             # Train model
    y_pred = model.predict(X_test_scaled)          # Predict on left-out subject

    # Store performance metrics
    fold_accuracies.append(accuracy_score(y_test, y_pred))
    fold_f1s.append(f1_score(y_test, y_pred, average='macro'))

    # Store predictions and ground truths for final summary
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# Print mean evaluation results across all folds
print(f"Mean LOSO Accuracy: {np.mean(fold_accuracies):.3f}")
print(f"Mean Macro F1-score: {np.mean(fold_f1s):.3f}")


# === Classification Report Heatmap ===
print("\nGenerating Classification Report Heatmap...")

report = classification_report(all_y_true, all_y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-3, :-1], annot=True, cmap="YlGnBu", cbar=True)
plt.title("Random Forest Classification Performance\nPrecision, Recall, F1 Score per Class")
plt.xlabel("Metric")
plt.ylabel("Activity Class")
plt.tight_layout()
plt.show()

# === Confusion Matrix ===
print("Generating Confusion Matrix...")

cm = confusion_matrix(all_y_true, all_y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix: Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# === Feature Importance Plot (Top 10) ===
print("Generating Feature Importance Plot...")

importances = model.feature_importances_
top_idx = np.argsort(importances)[-10:]  # Indices of top 10 features

plt.figure(figsize=(8, 5))
sns.barplot(x=importances[top_idx], y=[X.columns[i] for i in top_idx], orient='h')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
