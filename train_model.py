import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

print("--- 1. LOADING DATASET ---")
df = pd.read_csv("dataset/phishing_dataset.csv")

print("Dataset Shape (.shape):", df.shape)

print("\nFirst 5 Records (.head()):")
print(df.head())

print("\nRandom Records (.sample(3)):")
print(df.sample(3))

print("\n--- 2. DATA EXPLORATION & CLEANING ---")
print("\nStatistical Information (.describe()):")
print(df.describe())

print("\nMissing Values (.isnull().sum()):")
print(df.isnull().sum())

print("\nClass Distribution (.value_counts()):")
print(df["website_status"].value_counts())

print("\n--- 3. FEATURE SELECTION & EXACT 2K TEST SPLIT ---")
X = df.drop(columns=["website_status"])
y = df["website_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=2000, random_state=42, stratify=y
)
print(f"Training set rows: {X_train.shape[0]}")
print(f"Testing set rows: {X_test.shape[0]} (Exact 2,000 test records)")

print("\n--- 4. FEATURE SCALING ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("StandardScaler applied successfully (fit on training, transformed test)!")

print("\n--- 5. MODEL TRAINING ---")
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print("Logistic Regression model trained successfully!")

print("\n--- 6. MODEL EVALUATION ---")
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy Score:    {acc * 100:.2f}%")
print(f"Precision Score:   {precision * 100:.2f}%")
print(f"Recall Score:      {recall * 100:.2f}%")
print(f"F1-Score:          {f1 * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Legitimate (0)", "Suspicious (1)"]
)
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("PhishDetect AI - Confusion Matrix (2K Test Set)")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.close()

print("\n--- 7. SAVING TRAINED ASSETS ---")
joblib.dump(model, "models/phishing_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Saved model -> models/phishing_model.pkl")
print("Saved scaler -> models/scaler.pkl")
print("=== TRAINING PIPELINE COMPLETE ===")