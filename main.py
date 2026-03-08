# Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Classes
from visualization import Visualizer
from utils import ModelSaver
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

    # Histogram
    viz = Visualizer()
    viz.age_distribution(df)

    mt = ModelTrainer(random_state=42)
    mt.make_pipelines(X_train)
    mt.model_params()
    results = mt.hyperparameter_search(X_train, y_train)
    
    model_preds = {
        'rf':     (best_rf.predict(X_test),  best_rf.predict_proba(X_test)[:, 1]),
        'svm':    (best_svm.predict(X_test), best_svm.predict_proba(X_test)[:, 1]),
        'logreg': (best_lr.predict(X_test),  best_lr.predict_proba(X_test)[:, 1]),
    }
    
    for name, (y_pred, y_proba) in model_preds.items():
        print("\n" + "=" * 60)
        print(f"Classification Report — {name.upper()}")
        print("=" * 60)
        print(classification_report(y_test, y_pred, target_names=["No", "Yes"], digits=3))
        
    # Visualizations
    viz.confusion_matrix_lr(y_test, y_pred_lr)
    
    viz.roc_comparison(y_test,
    model_preds['rf'][1],
    model_preds['svm'][1],
    model_preds['logreg'][1])
    
    # Models in pickle
    ms = ModelSaver()
    ms.save_model(best_rf, RF_MODEL_PATH)
    ms.save_model(best_svm, SVM_MODEL_PATH)
    ms.save_model(best_lr, LR_MODEL_PATH)

if __name__ == "__main__":
    main()



