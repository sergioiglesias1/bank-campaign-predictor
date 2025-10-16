# Bank Marketing Campaign Predictions

This project applies machine learning techniques to predict whether a client will subscribe to a bank term deposit after a marketing call, using the UCI Bank Marketing Dataset.

## Project Overview
This repository compares different machine learning models to classify client responses (deposit vs no deposit).  
The models include:

- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **Logistic Regression**

> All with hyperparameter tuning via RandomizedSearchCV

The pipeline involves:

- Preprocessing categorical variables (label encoding)
- Feature scaling (for SVM and Logistic Regression)
- Training multiple classifiers
- Evaluating performance with accuracy, precision, recall, F1-score, and AUC-ROC
- Visualizing results with confusion matrices, ROC curves, feature importance, and boxplots

## Data Sources
- UCI Bank Marketing Dataset (`bank-additional-full.csv`)
- Link to data: [Kaggle Bank Marketing Dataset](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv/data)
- Contains client demographic data, call duration, campaign info, and economic indicators

All dataset files are stored in the `data/` folder.

## How It Works

### 1. Data Loading & Cleaning
- Load CSV from `data/` directory, sample 3-6% of data for faster training is a recommendation -> (df = df.sample(frac=0.03))
- Encode target variable (`yes` → 1, `no` → 0)

### 2. Exploratory Analysis
- Age distribution vs subscription -> We use 82 years as age limit to make a clearer visualization
- Data inspection and structure

### 3. Preprocessing
- Label encoding of categorical features
- Standard scaling of features for SVM and Logistic Regression pipelines with balanced data

### 4. Model Training
- Train/test split with stratification
- Pipelines are used for SVM and Logistic Regression to combine scaling + model
- All models are tuned with RandomizedSearchCV for optimal hyperparameters

> Random Forest is not in a pipeline because it does not need scaling.

### 5. Evaluation
- Accuracy, precision, recall, F1-score, and AUC-ROC
- Confusion matrices for misclassifications
- ROC curves for model comparison
- Random Forest feature importance (top 10 features)
- Boxplot: call duration by subscription result

## File Structure
```
.
├── data/                 # Dataset
├── models/               # The best models
├── plots/                # Generated plots (confusion matrices, ROC curves, feature importance...)
├── .gitignore
├── README.md
├── bank_ml_project.py    # Main training
└── requirements.txt      # Python dependencies
```

## Dependencies
- pandas
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
python3 bank_ml.py
```

Expected output:

```
=======================================================
PERFORMANCE METRICS PER MODEL
=======================================================
Model          Acc     Prec    Rec     F1      AUC     
-------------------------------------------------------
svm            0.851   0.426   0.931   0.585   0.943
logreg         0.865   0.452   0.911   0.604   0.944
rf             0.900   0.535   0.870   0.663   0.953
=======================================================
False Positives (wasted calls): 701
False Negatives (lost clients): 121
Predicted Acceptance Rate: 18.3%
Real Acceptance Rate: 11.3%
```

## Model Saving
- The best models are saved using `joblib` for future predictions, you can find it in the `models/` folder.
- To open a `.pkl` in python, copy this code:
```python
import joblib

for f in ["best_rf_model_grid.pkl", "best_svm_model_grid.pkl", "best_logreg_model_grid.pkl"]:
    model = joblib.load(f"models/{f}")
    print(f"\n{f}:")
    print(model)
    print(model.get_params())
```
## Future Improvements
- Experiment hyperparameter optimization with `GridSearchCV`
- Try Gradient Boosting or XGBoost
- Incorporate external macroeconomic indicators as features

## License
This project is licensed under the MIT License.
