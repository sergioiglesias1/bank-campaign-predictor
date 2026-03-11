# Packages
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Classes
from visualization import Visualizer
from utils import ModelSaver, ThresholdAnalyzer
from modeling import ModelTrainer

DATA_PATH = "data/cleaned_data.csv"
RF_MODEL_PATH = "models/best_rf_model.pkl"
SVM_MODEL_PATH = "models/best_svm_model.pkl"
LR_MODEL_PATH = "models/best_logreg_model.pkl"

def main():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found: {DATA_PATH}")
        return

    df['was_contacted'] = (df['pdays'] != 999).astype(int)
    df['pdays'] = df['pdays'].replace(999, 0)
    
    y = df['accepts'].map({'yes': 1, 'no': 0})
    X = df.drop('accepts', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    print(f"Training set: {X_train.shape}\nTesting set: {X_test.shape}\n")

    viz = Visualizer()
    viz.age_distribution(df)

    mt = ModelTrainer(random_state=42)
    mt.make_pipelines(X_train)
    mt.model_params()
    results = mt.hyperparameter_search(X_train, y_train)

    best_rf  = results['rf']['best_estimator']
    best_svm = results['svm']['best_estimator']
    best_lr  = results['logreg']['best_estimator']

    # Best AUC model probabilities
    probabilities = best_lr.predict_proba(X_test)[:, 1]

    # Threshold Table 
    ta = ThresholdAnalyzer()
    df_thresh = ta.sweep(y_test, probabilities)
    print("\n" + "=" * 60)
    print("Threshold Sweep — Logistic Regression")
    print("=" * 60)
    print(df_thresh.to_string(index=False))

    # Visualizations
    viz.roc_comparison(y_test,
        best_rf.predict_proba(X_test)[:, 1],
        best_svm.predict_proba(X_test)[:, 1],
        probabilities)

    # Models in pickle
    ms = ModelSaver()
    ms.save_model(best_rf, RF_MODEL_PATH)
    ms.save_model(best_svm, SVM_MODEL_PATH)
    ms.save_model(best_lr, LR_MODEL_PATH)

if __name__ == "__main__":
    main()
