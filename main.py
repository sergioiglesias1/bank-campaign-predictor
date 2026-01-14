# Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
# Classes
from visualization import Visualizer
from utils import ModelSaver
from modeling import ModelTrainer

def main():
    df = pd.read_csv(r"data/bank-additional-full.csv", sep=';')
    df = df.rename(columns={'y': 'accepts'})

    y = df["accepts"]
    X = df.drop("accepts", axis=1)

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

    # EDA
    print(df.describe())
    print(df.head(3))
    print(f"\nNull values per column:\n{df.isnull().sum()}")
    print(f"Dataset: {X.shape}")

    # Histogram
    viz = Visualizer()
    viz.age_distribution(df)

    mt = ModelTrainer(random_state=42)
    mt.make_pipelines(df)
    mt.model_params()
    results = mt.hyperparameter_search(X_train, y_train)

    # Best models
    best_rf = results['rf']['best_estimator']
    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
    
    best_svm = results['svm']['best_estimator']
    y_pred_svm = best_svm.predict(X_test)
    y_proba_svm = best_svm.predict_proba(X_test)[:, 1]
    
    best_lr = results['logreg']['best_estimator']
    y_pred_lr = best_lr.predict(X_test)
    y_proba_lr = best_lr.predict_proba(X_test)[:, 1]
    
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
    viz.confusion_matrix_rf(y_test, y_pred_rf)
    viz.roc_comparison(y_test, y_proba_rf, y_proba_svm, y_proba_lr)
    viz.feature_importance(best_rf)
    viz.call_duration_boxplot(df)

    # Models in pickle
    ms = ModelSaver()
    ms.save_model(best_rf, "best_rf_model.pkl")
    ms.save_model(best_svm, "best_svm_model.pkl")
    ms.save_model(best_lr, "best_logreg_model.pkl")

if __name__ == "__main__":
    main()
