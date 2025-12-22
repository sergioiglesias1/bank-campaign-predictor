import joblib
from sklearn.base import BaseEstimator

def save_model(model: BaseEstimator, path: str):
    joblib.dump(model, path)
    print(f"\nThe model has been saved successfully at: {path}\n")
