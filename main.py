# Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
# Classes
from visualization import Visualizer
from utils import ModelSaver
from modeling import ModelTrainer

def main():
    try:
        df = pd.read_csv("data/cleaned_data.csv")
    except FileNotFoundError:
        print("File not found")
    except ModuleNotFoundError:
        print("Package not found, please install it or change the environment")
    else:
        print("The CSV loading was successful")
    
    y = df['accepts']
    X = df.drop('accepts', axis=1)

    lenc = LabelEncoder()
    y = lenc.fit_transform(y)

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

    # Best models
    best_rf = results['rf']['best_estimator']
    y_pred_rf, y_proba_rf = best_rf.predict(X_test), best_rf.predict_proba(X_test)[:, 1]
    
    best_svm = results['svm']['best_estimator']
    y_pred_svm, y_proba_svm = best_svm.predict(X_test), best_svm.predict_proba(X_test)[:, 1]
    
    best_lr = results['logreg']['best_estimator']
    y_pred_lr, y_proba_lr = best_lr.predict(X_test), best_lr.predict_proba(X_test)[:, 1]
    
    # Classification Report
    for name, res in results.items():
        model = res["best_estimator"]
        y_pred = model.predict(X_test)

        print("\n" + "=" * 60)
        print(f"Classification Report — {name.upper()}")
        print("=" * 60)
        print(classification_report(
                y_test,
                y_pred,
                target_names=["No", "Yes"],
                digits=3
            )
        )
    
    # Visualizations
    viz.confusion_matrix_lr(y_test, y_pred_lr)
    viz.roc_comparison(y_test, y_proba_rf, y_proba_svm, y_proba_lr)
    viz.feature_importance(best_rf)

    # Models in pickle
    ms = ModelSaver()
    ms.save_model(best_rf, "models/best_rf_model.pkl")
    ms.save_model(best_svm, "models/best_svm_model.pkl")
    ms.save_model(best_lr, "models/best_logreg_model.pkl")

if __name__ == "__main__":
    main()

