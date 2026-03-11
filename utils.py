import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_recall_fscore_support, roc_curve

class ModelSaver:
    def save_model(self, model: BaseEstimator, path: str) -> str | None:
        try:
            joblib.dump(model, path)
            print(f"\nThe model has been saved successfully at: {path}\n")
            return path
        except Exception as e:
            print(f"[Error Saving Model] {e}")
            return None

class ThresholdAnalyzer:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    def sweep(self, y_test, probabilities) -> pd.DataFrame:
        fpr_arr, _, thresh_arr = roc_curve(y_test, probabilities)
        
        results = []
        for t in self.thresholds:
            y_pred = (probabilities >= t).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=0
            )
            idx = np.argmin(np.abs(thresh_arr - t))
            results.append({
                'threshold': t,
                'precision': round(precision, 3),
                'recall':    round(recall, 3),
                'f1':        round(f1, 3),
                'fpr':       round(fpr_arr[idx], 3)
            })

        return pd.DataFrame(results)
