# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
)

df = pd.read_csv(r"data/bank-additional-full.csv", sep=';')
df = df.rename(columns={'y': 'accepts'})  # yes/no -> clear name

# Study data
print(df.describe())
print(df.head(3))
print(f"\nNull values per column:\n{df.isnull().sum()}")

# Data visualization
plt.figure(figsize=(8,4))
sns.histplot(data=df, x='age', hue='accepts', common_norm=False, kde=True, fill=True, alpha=0.3)
plt.xticks(range(int(df['age'].min()), 81, 5))
plt.title('Age Distribution by Subscription')
plt.tight_layout()
plt.show()

# encoding binary target var
lbl_enc = LabelEncoder() # yes/no o true/false -> 0|1
df['accepts'] = lbl_enc.fit_transform(df['accepts']) # not get_dummies bc two classes and simplify train test
y = df['accepts']
X = df.drop('accepts', axis=1)
X_cod = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_cod, y, test_size=0.2, random_state=42, stratify=y # balances train test
)

# Dimensions
print(f"Dataset: {X.shape}")

# pipelines for svm and logreg
pipelines = {
    'svm': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(probability=True, random_state=42))
    ]),
    'logreg': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
}

# no pipeline here
models = {
    'rf': RandomForestClassifier(random_state=42, class_weight='balanced')
}

# hyperparameter assignment
param_grids = {
    'svm': {'model__C': [0.1, 1, 10], 'model__gamma': [0.01, 0.1, 1]},
    'logreg': {'model__C': [0.01, 0.1, 1, 10]},
    'rf': {'n_estimators': [100, 300, 500], 'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 8], 'min_samples_leaf': [1, 3, 5]}
}

# hyperparameter tuning
results = {}
for name, pipe in pipelines.items():
    search = RandomizedSearchCV(pipe, param_grids[name], cv=3, n_jobs=-1, verbose=1, scoring='roc_auc', random_state=42)
    search.fit(X_train, y_train)
    results[name] = {
        'best_score': search.best_score_,
        'best_params': search.best_params_,
        'best_estimator': search.best_estimator_
    }

# random forest
search_rf = RandomizedSearchCV(models['rf'], param_grids['rf'], cv=3, n_jobs=-1, verbose=1, scoring='roc_auc', random_state=42)
search_rf.fit(X_train, y_train)
results['rf'] = {
    'best_score': search_rf.best_score_,
    'best_params': search_rf.best_params_,
    'best_estimator': search_rf.best_estimator_
}

# best models
best_rf = results['rf']['best_estimator']
y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

best_svm = results['svm']['best_estimator']
y_pred_svm = best_svm.predict(X_test)
y_proba_svm = best_svm.predict_proba(X_test)[:, 1]

best_lr = results['logreg']['best_estimator']
y_pred_lr = best_lr.predict(X_test)
y_proba_lr = best_lr.predict_proba(X_test)[:, 1]

# classification report + auc roc
print("\n" + "=" * 55)
print("PERFORMANCE METRICS PER MODEL")
print("=" * 55)
print(f"{'Model': <15}{'Acc': <8}{'Prec': <8}{'Rec': <8}{'F1': <8}{'AUC': <8}")
print("-" * 55)

for name, res in results.items():
    model = res['best_estimator']
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"{name:<15}{acc:.3f}   {prec:.3f}   {rec:.3f}   {f1:.3f}   {auc:.3f}")

print("=" * 55)

# output visualizations
fig, axes = plt.subplots(2,2, figsize=(20, 12))
fig.suptitle('BANK MARKETING CAMPAIGN ANALYSIS', fontsize=24, fontweight='bold')

# plot 1
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(ax=axes[0,0], cmap='viridis')
axes[0,0].set_title('Random Forest Confusion Matrix', fontweight='bold')

# plot 2
RocCurveDisplay.from_predictions(y_test, y_proba_svm, ax=axes[0,1], name="SVM")
RocCurveDisplay.from_predictions(y_test, y_proba_rf, ax=axes[0,1], name="Random Forest")
RocCurveDisplay.from_predictions(y_test, y_proba_lr, ax=axes[0,1], name="LogReg")
axes[0,1].set_title('ROC Curve Comparison', fontweight='bold')
axes[0,1].legend()

# plot 3
feature_importance = pd.DataFrame({
    'feature': X_cod.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False).head(10)
sns.barplot(x='importance', y='feature', data=feature_importance, ax=axes[1,0], palette='viridis')
axes[1,0].set_title('Top 10 Feature Importance', fontweight='bold')

# plot 4
df_plot = df.copy()
df_plot['Subscription'] = df_plot['accepts'].map({0: 'No', 1: 'Yes'})
sns.boxplot(x='Subscription', y='duration', data=df_plot, ax=axes[1,1], palette='Set1')
axes[1,1].set_title('Call Duration by Subscription', fontweight='bold')
plt.show()

# save models
def save_model(model, path):
    joblib.dump(model, path)
    print(f"\nThe model has been saved successfully at: {path}\n")
    
save_model(best_rf, "best_rf_model_grid.pkl")
save_model(best_svm, "best_svm_model_grid.pkl")
save_model(best_lr, "best_logreg_model_grid.pkl")

# final output
rf_cm = confusion_matrix(y_test, y_pred_rf)
fp, fn = rf_cm[0,1], rf_cm[1,0]
print(f"False Positives (wasted calls): {fp}")
print(f"False Negatives (lost clients): {fn}")
print(f"Predicted Acceptance Rate: {(y_pred_rf.sum()/len(y_test))*100:.1f}%")
print(f"Real Acceptance Rate: {(y_test.mean()*100):.1f}%")