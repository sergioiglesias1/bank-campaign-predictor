# Bank Marketing Campaign Predictions

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

This project predicts which bank clients are likely to subscribe to a term deposit after a marketing call, with a focus on maximizing recall while controlling campaign costs.

## Project Overview
This repository compares different machine learning models to classify client responses (deposit vs no deposit).  

The models include:

- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **Logistic Regression**

## Why these models?
- **Logistic Regression**: Stable, interpretable, good for threshold-based decisions.  
- **SVM**: Maximizes recall when missing a subscriber is costly.  
- **Random Forest**: Non-linear benchmark and feature importance analysis.

## Pipeline Overview

- **Data**: Load dataset, encode target (`yes` → 1, `no` → 0), train/test split with stratification  
- **Preprocessing & Architecture**: OOP classes (`ModelTrainer`, `Visualizer`, `ModelSaver`); numerical features scaled, categorical one-hot encoded  
- **Modeling**: Pipelines for SVM, Logistic Regression, Random Forest; hyperparameter tuning with `RandomizedSearchCV` (ROC-AUC)  
- **Evaluation**: Metrics (Accuracy, Precision, Recall, F1, AUC-ROC) and plots (Confusion Matrix, ROC, Feature Importance, call duration analysis)

## Data Sources
- UCI Bank Marketing Dataset (`bank-additional-full.csv`)
- [Dataset link](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv/data)  
- Contains client demographic data, call duration, campaign info, and economic indicators

All dataset files are stored in the `data/` folder.

## How It Works
- Run `EDA.ipynb` first to analyze the data, detect outliers, and generate `data/cleaned_data.csv`.
- Run `main.py` which loads the cleaned data.
- The script applies scaling/encoding via pipelines and trains SVM, Logistic Regression, and Random Forest.
- The main file runs all the python files with classes and functions to train the models and save them in the `models/` folder.

### Evaluation Metrics
- Accuracy, precision, recall, F1-score, and AUC-ROC
- Confusion matrices for misclassifications
- ROC curves for model comparison
- Random Forest feature importance (top 10 features)

## Results & Model Performance

### Dataset Overview
- **Total samples**: 41,189
- **Number of features**: 20
- **Positive class (subscription)**: **11.3%**

---

### Best Models After Hyperparameter Tuning (ROC-AUC)

| Model | Best ROC-AUC | Best Hyperparameters |
|------|-------------|----------------------|
| Random Forest | 0.953 | `n_estimators = 100`, `max_depth = 20`, `min_samples_leaf = 1` |
| SVM (RBF Kernel) | 0.943 | `C = 1`, `gamma = 0.01` |
| **Logistic Regression** | 0.944 | `C = 0.1` |

---

### Test Set Performance with focus on subscribed clients (Positive Class)

| Model              | Accuracy | Precision | Recall | F1-score |
|--------------------|:--------:|:---------:|:------:|:--------:|
| Random Forest       | 0.900 | 0.650 | 0.277  | 0.388    |
| SVM                 | 0.879    | 0.481     | 0.830  | 0.609    |
| Logistic Regression | 0.891    | 0.514    | 0.787 | 0.622 |

> Although SVM achieves higher recall, Logistic Regression provides a better precision–recall tradeoff, resulting in fewer wasted calls per captured subscriber.

---

### Notes on Model Interpretation

- **High Accuracy vs Low Recall**: Even with `class_weight='balanced'` and stratified splits, Random Forest reaches 90% accuracy but misses many subscribed clients. This is due to the default threshold; adjusting it or optimizing for recall improves detection of subscribers.

- **Business Context of Errors**:  
  - **Wasted calls (Type I errors)**: calls made to clients who would not subscribe  
  - **Lost clients (Type II errors)**: potential subscribers not predicted by the model  

Here the focus should be on maximizing recall for positive clients, since missing a potential subscriber is more costly than an extra call.

---

### Business Impact Summary
- **Wasted calls (Type I Error)**: 701
- **Lost clients (Type II Error)**: 121
- **Predicted Acceptance Rate**: 18.3%  
- **Real Acceptance Rate**: 11.3%

> Computed on the test set using Logistic Regression, because it provides the best business-aligned performance.
---

### Essential Point

**The simplest model (Logistic Regression) delivers the best business-aligned performance**, even though SVM has a better recall, Logistic Regression has a lot more precision and F1, so it captures more positive cases while maintaining a better balance between missed positives and false alarms.

---

## File Structure
```
.
├── data/
│   ├── bank-additional-full.csv
│   └── cleaned_data.csv
├── models/
│   ├── best_logreg_model.pkl
│   ├── best_rf_model.pkl
│   └── best_svm_model.pkl
├── .gitignore
├── EDA.ipynb
├── LICENSE
├── main.py
├── modeling.py
├── README.md
├── requirements.txt
├── utils.py
└── visualization.py
```

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

## Usage
Run the script with:

```bash
python3 main.py
```

## Model Saving
- The best models are saved using `joblib` for future predictions, you can find it in the `models/` folder.
- To open a `.pkl` in python, copy this code:
```python
import joblib

for f in ["best_rf_model.pkl", "best_svm_model.pkl", "best_logreg_model.pkl"]:
    model = joblib.load(f"models/{f}")
    print(f"\n{f}:")
    print(model)
    print(model.get_params())
```
## Future Improvements
- Experiment hyperparameter optimization with `GridSearchCV`
- Try Gradient Boosting or XGBoost
- Add macroeconomical variables to the dataset

## License
This project is licensed under the MIT License.
