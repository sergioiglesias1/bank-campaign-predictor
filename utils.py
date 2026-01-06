import joblib
from sklearn.base import BaseEstimator

class ModelSaver:
    def save_model(
        self, model: BaseEstimator, 
        path: str
    ):
        joblib.dump(model, path)
        print(f"\nThe model has been saved successfully at: {path}\n")
