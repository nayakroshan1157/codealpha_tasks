import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings("ignore")

print("🔹 Training Models...\n")

# =========================
# 🩸 1. DIABETES MODEL
# =========================

diabetes = pd.read_csv("datasets/diabetes.csv")

# Encode categorical columns
cat_col = diabetes.select_dtypes(include='object').columns
for col in cat_col:
    diabetes[col] = diabetes[col].astype('category').cat.codes

X = diabetes.drop('diabetes', axis=1)
y = diabetes['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

diabetes_model = LogisticRegression(max_iter=1000)
diabetes_model.fit(X_train, y_train)

pred = diabetes_model.predict(X_test)
print("✅ Diabetes Accuracy:", accuracy_score(y_test, pred))

pickle.dump(diabetes_model, open("diabetes_model.pkl", "wb"))


# =========================
# ❤️ 2. HEART DISEASE MODEL
# =========================

heart = pd.read_csv("datasets/hearts.csv")

X = heart.drop('target', axis=1)
y = heart['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

heart_model = LogisticRegression(max_iter=1000)
heart_model.fit(X_train, y_train)

pred = heart_model.predict(X_test)
print("✅ Heart Disease Accuracy:", accuracy_score(y_test, pred))

pickle.dump(heart_model, open("heart_model.pkl", "wb"))


# =========================
# 🎗 3. BREAST CANCER MODEL
# =========================

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

breast_model = LogisticRegression(max_iter=1000)
breast_model.fit(X_train, y_train)

pred = breast_model.predict(X_test)
print("✅ Breast Cancer Accuracy:", accuracy_score(y_test, pred))

pickle.dump(breast_model, open("breast_cancer_model.pkl", "wb"))

print("\n🎉 All Models Trained and Saved Successfully!")
