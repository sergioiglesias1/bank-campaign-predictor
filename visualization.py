import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline

def plot_eda(df):
    plt.figure(figsize=(8,4))
    sns.histplot(data=df, x='age', hue='accepts', common_norm=False, kde=True, fill=True, alpha=0.3)
    plt.xticks(range(int(df['age'].min()), 81, 5))
    plt.title('Age Distribution by Subscription')
    plt.tight_layout()
    plt.show()

def plot_results(
    y_test: pd.Series, 
    y_pred_rf: np.ndarray, 
    y_proba_rf: np.ndarray, 
    y_pred_svm: np.ndarray, 
    y_proba_svm: np.ndarray, 
    y_pred_lr: np.ndarray, 
    y_proba_lr: np.ndarray, 
    best_rf: Pipeline, 
    df: pd.DataFrame):

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
    
    # plot 3, complex due to feature names
    try:
        rf_model = best_rf.named_steps['model']
        preprocessor = best_rf.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        
        if len(feature_names) == len(rf_model.feature_importances_):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            sns.barplot(x='importance', y='feature', data=feature_importance, ax=axes[1,0], palette='viridis')
            axes[1,0].set_title('Top 10 Feature Importance', fontweight='bold')
        else:
            axes[1,0].text(0.5, 0.5, 'Error: Feature length mismatch', ha='center')
            print(f"Mismatch: {len(feature_names)} features vs {len(rf_model.feature_importances_)} importances")
    except Exception as exc:
        axes[1,0].text(0.5, 0.5, f"Error: {str(exc)}")
        print(f"Plot 3 Error: {exc}")
    
    # plot 4
    df_plot = df.copy()
    df_plot['Subscription'] = df_plot['accepts'].map({0: 'No', 1: 'Yes'})
    sns.boxplot(x='Subscription', y='duration', data=df_plot, ax=axes[1,1], palette='Set1')
    axes[1,1].set_title('Call Duration by Subscription', fontweight='bold')
    plt.show()
