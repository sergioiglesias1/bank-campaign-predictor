# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay

df = pd.read_csv(r"C:\Users\Usuario\Downloads\messy_databases\bank-additional-full.csv", sep=';')

# View data
print(df.info())
print(df.head(3))
print(f"\nNull values per column:\n{df.isnull().sum()}") # nulls per column

lbl_enc = LabelEncoder() # yes/no o true/false -> 0|1
df['y'] = lbl_enc.fit_transform(df['y']) # not get_dummies bc two classes and simplify train test
y = df['y']
X = df.drop('y', axis=1)

# age/subscription
plt.figure(figsize=(7,3))
sns.swarmplot(y='age', x='y', data=df, palette='Set1', hue='y', legend=False)
plt.title('Age vs Subscription')
plt.xticks([0,1], ['No', 'Yes'])
plt.xlabel('Subscribed?')
plt.ylabel('Age')

X_codif = pd.get_dummies(X, drop_first=True) # binary -> dummies
feature_names = X.columns.tolist()
target_names = ['No Deposit', 'Deposit']
print(f"\nX with dummies:\n{X_codif.head(2)}")

# Train-Test
X_train, X_test, y_train, y_test = train_test_split(
    X_codif, y, test_size=0.2, random_state=42, stratify=y # stratify -> balances train test
)

print(f"Dataset Dimensions: {X.shape}")
print(f"Subscribed Clients: {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")

# Scaling X, train and test
scaler = StandardScaler() # normalize
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM
svm_model = SVC(kernel="rbf", C=1, gamma=0.01, probability=True, random_state=42)
# After trying different methods, changing C and gamma, I obtained the best results with C=1 and gamma=0.01
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

# Random Forest
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_proba_lr = lr_model.predict_proba(X_test_scaled)[:,1]

# We define the three models, to compare them later
models = {
    'SVM': (y_pred_svm, y_pred_proba_svm),
    'Random Forest': (y_pred_rf, y_proba_rf),
    'LogReg': (y_pred_lr, y_proba_lr)
}

print("\n" + "=" * 70)
print("PERFORMANCE METRICS PER MODEL")
print("=" * 70)

print(f"{'Model': <15}{'Accuracy': <10} {'Precision': <10}  {'Recall': <10}  {'F1-Score': <10}   {'AUC-ROC': <10}")
print("-" * 70)

# We set the accuracy metrics
for name, (pred, proba) in models.items():
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
   
    print(f"{name:<15}  {acc:.3f}       {prec:.3f}      {rec:.3f}       {f1:.3f}        {auc:.3f}")
    
print("=" * 70)

# Defining the plots of the output
fig, axes = plt.subplots(2,2, figsize=(20, 12))
fig.suptitle('BANK MARKETING CAMPAIGN ANALYSIS', fontsize=24, fontweight='bold')

# Plot nº1
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=axes[0,0], cmap='viridis')
axes[0,0].set_title('Random Forest Confusion Matrix', fontweight='bold')

# Plot nº2
auc_svm = roc_auc_score(y_test, y_pred_proba_svm)
auc_rf = roc_auc_score(y_test, y_proba_rf)
auc_lr = roc_auc_score(y_test, y_proba_lr)

RocCurveDisplay.from_predictions(y_test, y_pred_proba_svm, ax=axes[0,1], name=f"SVM")
RocCurveDisplay.from_predictions(y_test, y_proba_rf, ax=axes[0,1], name=f"Random Forest")
RocCurveDisplay.from_predictions(y_test, y_proba_lr, ax=axes[0,1], name=f"Logistic Regression")

axes[0,1].set_title('ROC Curve Comparison', fontweight='bold')
axes[0,1].legend()

# Plot nº3
X_codif = pd.get_dummies(X, drop_first=True) # no multicolinelidad
real_feature_names = X_codif.columns.tolist()
feature_importance = pd.DataFrame({
    'feature': real_feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(8)

sns.barplot(x='importance', y='feature', data=feature_importance, ax=axes[1,0], hue='importance', palette='viridis')
axes[1,0].set_title('Top 10 Feature Importance', fontweight='bold')
axes[1,0].set_xlabel('Importance')

# Plot nº4
df_plot = df.copy()
df_plot['Subscription'] = df_plot['y'].map({0: 'No', 1: 'Yes'})
sns.boxplot(x='Subscription', y='duration', data=df_plot, ax=axes[1,1], hue='Subscription', palette='Set1')
axes[1,1].set_title('Call Duration by Subscription Result', fontweight='bold')
axes[1,1].set_ylabel('Call Duration (seconds)')
axes[1,1].set_xlabel('Subscription')
plt.show()

# Clients and rates
print(f"Identified clients: {y_pred_rf.sum()}/{len(y_test)}")
print(f"Predicted Rate: {(y_pred_rf.sum()/len(y_test))*100:.1f}%")
print(f"Real Rate: {(y_test.sum()/len(y_test))*100:.1f}%")

# False positives and false negatives
rf_cm = confusion_matrix(y_test, y_pred_rf)
fp = rf_cm[0, 1]
fn = rf_cm[1, 0]
print(f"False Positives: {fp} (wasted calls)")
print(f"False Negatives: {fn} (lost clients)")
