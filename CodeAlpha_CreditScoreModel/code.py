# ============================
# CREDIT SCORING ML PROJECT (FULL)
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Load Dataset
data = pd.read_csv("credit_data.csv")

# 2. Handle Missing Values
data.fillna(method='ffill', inplace=True)

# 3. Encode Categorical Columns
le_business = LabelEncoder()
le_service = LabelEncoder()

data['Business_Type'] = le_business.fit_transform(data['Business_Type'])
data['Service_Sector'] = le_service.fit_transform(data['Service_Sector'])

# 4. Feature & Target Split
X = data.drop('Credit_Status', axis=1)
y = data['Credit_Status']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train Models
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 8. Predictions
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:,1]

y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:,1]

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

# 9. Evaluation Function
def evaluate_model(name, y_test, y_pred, y_prob):
    print(f"\n{name} Evaluation")
    print(classification_report(y_test, y_pred))
    roc = roc_auc_score(y_test, y_prob)
    print("ROC AUC:", roc)
    return roc

roc_lr = evaluate_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
roc_dt = evaluate_model("Decision Tree", y_test, y_pred_dt, y_prob_dt)
roc_rf = evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)

# 10. ROC Curve Plot
plt.figure()
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={roc_lr:.2f})")
plt.plot(fpr_dt, tpr_dt, label=f"DT (AUC={roc_dt:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={roc_rf:.2f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# 11. Feature Importance (Random Forest)
importances = rf.feature_importances_
features = X.columns
fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
print("\nFeature Importances (Random Forest):")
print(fi_df.sort_values(by="Importance", ascending=False))

# ============================
# 🔹 USER INPUT & PREDICTION
# ============================

print("\n--- Enter Customer Details ---")

income = float(input("Enter Income: "))
business_type = input("Enter Business Type (Salaried / Self-Employed / Business Owner): ")
service_sector = input("Enter Service Sector (IT / Finance / Healthcare / Education / Retail): ")
debts = float(input("Enter Total Debts: "))
age = int(input("Enter Age: "))
payment_history = float(input("Enter Payment History Score (0–100): "))
credit_util = float(input("Enter Credit Utilization (0.1 – 0.9): "))
loan_amt = float(input("Enter Loan Amount: "))
tenure = int(input("Enter Loan Tenure (months): "))

# Encode categorical inputs
business_type_enc = le_business.transform([business_type])[0]
service_sector_enc = le_service.transform([service_sector])[0]

# Create input array
user_data = np.array([[income, business_type_enc, service_sector_enc, debts, age,
                       payment_history, credit_util, loan_amt, tenure]])

# Scale input
user_data_scaled = scaler.transform(user_data)

# Predict using Random Forest (best model)
prediction = rf.predict(user_data_scaled)[0]
probability = rf.predict_proba(user_data_scaled)[0][1]

# Output
print("\n--- Prediction Result ---")
if prediction == 1:
    print("⚠ Credit Status: DEFAULT RISK")
else:
    print("✅ Credit Status: GOOD CUSTOMER")

print(f"Default Probability: {probability:.2f}")
