# packages
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, 
    f1_score, roc_auc_score
)

# other two archves
from visualization import plot_eda, plot_results
from utils import save_model

def main():
    df = pd.read_csv(r"data/bank-additional-full.csv", sep=';')
    df = df.rename(columns={'y': 'accepts'})
    
    # study df
    print(df.describe())
    print(df.head(3))
    print(f"\nNull values per column:\n{df.isnull().sum()}")
    
    # introductory histogram
    plot_eda(df)
    
    # encoding binary target var
    lbl_enc = LabelEncoder() # yes|no or true|false -> 1|0
    df['accepts'] = lbl_enc.fit_transform(df['accepts'])
    y = df['accepts']
    X = df.drop('accepts', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset: {X.shape}")

    # Preprocessor -> scaler for svm & lr and encoder for rf
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), selector(dtype_exclude=object)),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), selector(dtype_include=object))
        ],
        remainder='passthrough'
    )

    pipelines = {
        'svm': Pipeline([
            ('preprocessor', preprocessor),
            ('model', SVC(probability=True, random_state=42))
        ]),
        'logreg': Pipeline([
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ]),
        'rf': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])
    }
    
    # hyperparameters
    param_grids = {
        'svm': {'model__C': [0.1, 1, 10], 'model__gamma': [0.01, 0.1, 1]},
        'logreg': {'model__C': [0.01, 0.1, 1, 10]},
        'rf': {
            'model__n_estimators': [100, 300], 
            'model__max_depth': [5, 10, 20],
            'model__min_samples_leaf': [1, 3]
        }
    }
    
    # hyperparameter tuning
    results = {}
    for name, pipe in pipelines.items():
        search = RandomizedSearchCV(
            pipe, param_grids[name], cv=3, n_jobs=-1, verbose=1, scoring='roc_auc', random_state=42
        )
        search.fit(X_train, y_train)
        results[name] = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'best_estimator': search.best_estimator_
        }
        print(f"Best AUC for {name}: {search.best_score_:.3f}")

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
    
    # classification report manually bc the normal one does not have AUC-ROC
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
    plot_results(y_test, y_pred_rf, y_proba_rf, y_pred_svm, y_proba_svm, y_pred_lr, y_proba_lr, best_rf, df)
    
    # save models in pickle
    save_model(best_rf, "best_rf_model_grid.pkl")
    save_model(best_svm, "best_svm_model_grid.pkl")
    save_model(best_lr, "best_logreg_model_grid.pkl")
    
    # a confusion matrix to see the fp, fn, tp, tn
    rf_cm = confusion_matrix(y_test, y_pred_rf)
    fp, fn = rf_cm[0,1], rf_cm[1,0]
    print(f"False Positives (wasted calls): {fp}")
    print(f"False Negatives (lost clients): {fn}")
    print(f"Predicted Acceptance Rate: {(y_pred_rf.sum()/len(y_test))*100:.1f}%")
    print(f"Real Acceptance Rate: {(y_test.mean()*100):.1f}%")

if __name__ == "__main__":
    main()
