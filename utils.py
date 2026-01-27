import joblib
from sklearn.base import BaseEstimator

class ModelSaver:
    def save_model(self, model: BaseEstimator, path: str) -> str | None:
        try:
            joblib.dump(model, path)
            print(f"\nThe model has been saved successfully at: {path}\n")
            return path
        except Exception as e:
            print(f"[Error Saving Model] {e}")
            return None