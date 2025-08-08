import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)
import xgboost as xgb
import joblib

os.makedirs('models', exist_ok=True)
df = pd.read_csv("/Users/sidhaanthkapoor/Desktop/AI-Driven Employee Performance & Attrition Predictor/data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

X = df[["MonthlyIncome", "Age", "EmployeeNumber", "TotalWorkingYears",
        "DailyRate", "MonthlyRate", "OverTime", "DistanceFromHome", "HourlyRate"]]
y = df["Attrition"].map({'Yes': 1, 'No': 0})

if X["OverTime"].dtype == 'object':
    X["OverTime"] = X["OverTime"].map({'Yes': 1, 'No': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
y_probs = xgb_model.predict_proba(X_test)[:, 1]

print("XGBoost Model Performance on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['No Attrition', 'Yes Attrition']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

precision, recall, _ = precision_recall_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for XGBoost')
plt.tight_layout()
pr_path = os.path.join('models', 'xgboost_precision_recall_curve.png')
plt.savefig(pr_path)
print(f"Precision-recall curve saved as '{pr_path}'")

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGBoost')
plt.legend(loc='lower right')
plt.tight_layout()
roc_path = os.path.join('models', 'xgboost_roc_curve.png')
plt.savefig(roc_path)
print(f"ROC curve saved as '{roc_path}'")

model_path = os.path.join('models', 'xgboost_model.pkl')
joblib.dump(xgb_model, model_path)
print(f"XGBoost model saved as '{model_path}'")