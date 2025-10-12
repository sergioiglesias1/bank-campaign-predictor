# Bank Marketing Campaign Predictions

This project applies machine learning techniques to predict whether a client will subscribe to a bank term deposit after a marketing call, using the UCI Bank Marketing Dataset.

## Project Overview
This repository compares different machine learning models to classify client responses (deposit vs no deposit).  
The models include:

- **Support Vector Machine (SVM)** with RBF kernel
- **Random Forest Classifier** (with hyperparameter tuning via GridSearchCV)
- **Logistic Regression**

The pipeline involves:

- Preprocessing categorical variables (label encoding + one-hot encoding)
- Feature scaling (for SVM and Logistic Regression)
- Training multiple classifiers
- Evaluating performance with accuracy, precision, recall, F1-score, and AUC-ROC
- Visualizing results with confusion matrices, ROC curves, feature importance, and boxplots

## Data Sources
- UCI Bank Marketing Dataset (`bank-additional-full.csv`)
- Link to data: [Kaggle Bank Marketing Dataset](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv/data)
- Contains client demographic data, call duration, campaign info, and economic indicators

All dataset files are stored in the `data/` folder (not included in the repo).

## How It Works

### 1. Data Loading & Cleaning
- Load CSV from `data/` directory, sample 3-6% of data for faster training is a recommendation
- Encode target variable (`yes` → 1, `no` → 0)

### 2. Exploratory Analysis
- Age distribution vs subscription
- Call duration by subscription

### 3. Preprocessing
- One-hot encoding of categorical features
- Standard scaling of features for SVM and Logistic Regression pipelines

### 4. Model Training
- Train/test split with stratification
- Pipelines are used for SVM and Logistic Regression to combine scaling + model
- Random Forest is tuned with GridSearchCV for optimal hyperparameters, it is the most sensitive one

> GridSearch is applied only to Random Forest as it is more sensitive to hyperparameters.  
> SVM and Logistic Regression are left with standard configurations to maintain stability and reduce computation time.

### 5. Evaluation
- Accuracy, precision, recall, F1-score, and AUC-ROC
- Confusion matrices for misclassifications
- ROC curves for model comparison
- Random Forest feature importance (top 10 features)
- Boxplot: call duration by subscription result

## File Structure
```
.
├── bank_ml.py            # Main training + evaluation script
├── data/                 # Dataset
├── plots/                # Generated plots (confusion matrices, ROC curves, feature importance, boxplots)
├── requirements.txt      # Python dependencies
└── README.md
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

Expected output (numbers are indicative):

```
======================================================================
PERFORMANCE METRICS PER MODEL
======================================================================
Model          Acc     Prec    Rec     F1      AUC     
----------------------------------------------------------------------
SVM            0.914   0.718   0.385   0.501   0.942
Random Forest  0.902   0.542   0.860   0.665   0.953
LogReg         0.865   0.452   0.912   0.604   0.944
======================================================================
False Positives (wasted calls): 674
False Negatives (lost clients): 130
Predicted Acceptance Rate: 17.9%
Real Acceptance Rate: 11.3%
```

## Model Saving
The best Random Forest model is saved as `best_rf_model_grid.pkl` using `joblib` for future predictions.

## Future Improvements
- Experiment with `RandomizedSearchCV` for high-dimensional datasets
- Try Gradient Boosting or XGBoost
- Incorporate external macroeconomic indicators as features

## License
This project is licensed under the MIT License.
