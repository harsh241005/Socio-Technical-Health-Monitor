import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay, classification_report
import shap
import joblib

# 1. Load the clean dataset
df = pd.read_csv("data/processed/isa3_enriched_dataset.csv")

# ONLY the pure communication/sentiment features
safe_features = [
    'email_count_per_ticket', 'subject_length',                      
    'avg_sentiment', 'sentiment_variance', 'sentiment_trend',        
    'priority_numeric', 'has_enough_emails'                          
]

X = df[safe_features]
y = df['is_stalled']

# 2. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Train the Final Honest XGBoost Model
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
final_model = XGBClassifier(scale_pos_weight=scale_weight, random_state=42, eval_metric='logloss')
final_model.fit(X_train, y_train)

print("\n=== FINAL HONEST MODEL PERFORMANCE (Threshold 0.5) ===")
y_pred = final_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Active', 'Stalled']))

# Plot Precision-Recall Curve 
y_probs = final_model.predict_proba(X_test)[:, 1]
display = PrecisionRecallDisplay.from_predictions(y_test, y_probs, name="XGBoost (Honest Features)")
plt.title("ISA III: Precision-Recall Curve (Pure Communication Signals)")
plt.savefig("visuals/isa3_honest_pr_curve.png", bbox_inches='tight')
print("Saved PR Curve to visuals/isa3_honest_pr_curve.png")
plt.close()

print("\n=== GENERATING SHAP EXPLANATIONS ===")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

# Global Feature Importance
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary: Pure Socio-Technical Communication Signals")
plt.savefig("visuals/isa3_honest_shap_summary.png", bbox_inches='tight')
print("Saved SHAP Summary Plot to visuals/isa3_honest_shap_summary.png")
plt.close()

# Export for Streamlit Dashboard
joblib.dump(final_model, 'models/isa3_xgboost_honest.pkl')
joblib.dump(explainer, 'models/isa3_shap_explainer_honest.pkl')
print("\nSaved Model and Explainer to /models/ directory.")