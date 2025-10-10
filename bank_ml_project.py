# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
)

df = pd.read_csv(r"/kaggle/input/bankadditionalfullcsv/bank-additional-full.csv", sep=';')
df = df.rename(columns={'y': 'accepts'})  # yes/no -> clear name
df = df.sample(frac=0.06, random_state=42)
print(df.info())
print(df.head(4))
print(f"\nNull values per column:\n{df.isnull().sum()}")

# basic visualization
plt.figure(figsize=(8,4))
sns.histplot(data=df, x='age', hue='accepts', common_norm=False, kde=True, fill=True, alpha=0.3)
plt.title('Age Distribution by Subscription')
plt.tight_layout()
plt.show()

# encoding
lb_enc = LabelEncoder()
df['accepts'] = lb_enc.fit_transform(df['accepts'])

y = df['accepts']
X = df.drop('accepts', axis=1)
X_cod = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_cod, y, test_size=0.2, random_state=42, stratify=y # balances train test
)

print(f"Dataset: {X.shape}")

# SVM Pipeline
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel="rbf", C=1, gamma=0.01, probability=True, random_state=42))
])
svm_pipeline.fit(X_train, y_train)
y_pred_svm = svm_pipeline.predict(X_test)
y_proba_svm = svm_pipeline.predict_proba(X_test)[:, 1] # :,1 -> 'yes'

# Random Forest + GridSearch
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
# balance entre sensibilidad y overfitting con un GridSearch sobre hiperparámetros del rf
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3],
}
grid_rf = GridSearchCV(rf_model, param_grid, cv=2, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print(f"\nBest RF Params: {grid_rf.best_params_}")

y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

# Logistic Regression Pipeline
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
]) # needs scaling
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

# gridSearch only to rf because it's more sensitive to hyperparameters
# if used in all models -> higher computation time

models = {
    'SVM': (y_pred_svm, y_proba_svm),
    'Random Forest': (y_pred_rf, y_proba_rf),
    'LogReg': (y_pred_lr, y_proba_lr)
}

print("\n" + "=" * 70)
print("PERFORMANCE METRICS PER MODEL")
print("=" * 70)
print(f"{'Model': <15}{'Acc': <8}{'Prec': <8}{'Rec': <8}{'F1': <8}{'AUC': <8}")
print("-" * 70)

for name, (pred, proba) in models.items():
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    print(f"{name:<15}{acc:.3f}   {prec:.3f}   {rec:.3f}   {f1:.3f}   {auc:.3f}")

print("=" * 70)

# output visualizations
fig, axes = plt.subplots(2,2, figsize=(20, 12))
fig.suptitle('BANK MARKETING CAMPAIGN ANALYSIS', fontsize=24, fontweight='bold')

cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(ax=axes[0,0], cmap='viridis')
axes[0,0].set_title('Random Forest Confusion Matrix', fontweight='bold')

RocCurveDisplay.from_predictions(y_test, y_proba_svm, ax=axes[0,1], name="SVM")
RocCurveDisplay.from_predictions(y_test, y_proba_rf, ax=axes[0,1], name="Random Forest")
RocCurveDisplay.from_predictions(y_test, y_proba_lr, ax=axes[0,1], name="LogReg")
axes[0,1].set_title('ROC Curve Comparison', fontweight='bold')
axes[0,1].legend()

feature_importance = pd.DataFrame({
    'feature': X_cod.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False).head(10)
sns.barplot(x='importance', y='feature', data=feature_importance, ax=axes[1,0], palette='viridis')
axes[1,0].set_title('Top 10 Feature Importance', fontweight='bold')

df_plot = df.copy()
df_plot['Subscription'] = df_plot['accepts'].map({0: 'No', 1: 'Yes'})
sns.boxplot(x='Subscription', y='duration', data=df_plot, ax=axes[1,1], palette='Set1')
axes[1,1].set_title('Call Duration by Subscription', fontweight='bold')
plt.show()

def save_model(model, path):
    joblib.dump(model, path)
    print(f"\nThe model has been saved successfully at: {path}")

save_model(best_rf, "best_rf_model_grid.pkl")

# final output
rf_cm = confusion_matrix(y_test, y_pred_rf)
fp, fn = rf_cm[0,1], rf_cm[1,0]
print(f"False Positives (wasted calls): {fp}")
print(f"False Negatives (lost clients): {fn}")
print(f"Predicted Acceptance Rate: {(y_pred_rf.sum()/len(y_test))*100:.1f}%")
print(f"Real Acceptance Rate: {(y_test.mean()*100):.1f}%")
