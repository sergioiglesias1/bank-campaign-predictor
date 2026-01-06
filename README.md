# Bank Marketing Campaign Predictions

This project applies machine learning techniques to predict whether a client will subscribe to a bank term deposit after a marketing call, using the UCI Bank Marketing Dataset.

## Project Overview
This repository compares different machine learning models to classify client responses (deposit vs no deposit).  
The models include:

- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **Logistic Regression**

> All with hyperparameter tuning via RandomizedSearchCV

### Why these models?
- **Logistic Regression**: Serves as a strong base for binary classification problems. It provides excellent interpretability, allowing us to understand the impact of each feature on the probability of subscription.
- **Support Vector Machine (SVM)**: Effective in high-dimensional spaces. It attempts to find the optimal hyperplane that maximizes the margin between the two classes.
- **Random Forest**: A robust method that drives non-linear relationships well and it is a model with less likely to overfitting than individual decision trees. It also provides valuable data for feature importance.

### Pipeline
The pipeline involves:

1. **Data Loading & Target Encoding**:
   - Loading the dataset and encoding the target variable (`yes` → 1, `no` → 0) using `LabelEncoder`.
   - Splitting data into training and testing sets with stratification to maintain class balance.

2. **Object-Oriented Architecture**:
   The project follows an OOP design with dedicated classes for each responsibility:
   - `ModelTrainer` class (**modeling.py**): Handles all machine learning operations including pipeline creation, hyperparameter tuning, and model training.
   - `Visualizer` class (**visualization.py**): Manages all plotting and visualization functionalities with consistent styling.
   - `ModelSaver` class (**utils.py**): Provides utilities for model serialization and persistence.

2. **Preprocessing (ColumnTransformer)**:
   - **Numerical Features**: Applied `StandardScaler` to normalize distributions.
   - **Categorical Features**: Applied `OneHotEncoder` (dropping the first category to avoid multicollinearity) to convert categories into numerical vectors.

3. **Model Training & Tuning**:
   - Integrated preprocessing and model initialization into a single `Pipeline` object for each classifier (SVM, Logistic Regression, Random Forest).
   - Performed Hyperparameter Optimization using `RandomizedSearchCV` (for ROC-AUC).

4. **Evaluation & Visualization**:
   - Metrics: Accuracy, Precision, Recall, F1-score, and AUC-ROC.
   - Plots: Confusion Matrix, ROC Curves, Feature Importance (RF), a boxplot (with the duration of the calls depending if subscribed or not), and a histogram.

## Data Sources
- UCI Bank Marketing Dataset (`bank-additional-full.csv`)
- Link to data: [Kaggle Bank Marketing Dataset](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv/data)
- Contains client demographic data, call duration, campaign info, and economic indicators

All dataset files are stored in the `data/` folder.

## How It Works

### 1. Data Loading & Cleaning
- Load CSV from `data/` directory, sample 5-10% of data for faster training is a recommendation -> (df = df.sample(frac=0.05))
- Encode target variable (`yes` → 1, `no` → 0)

### 2. Exploratory Analysis
- Age distribution vs subscription
- Data inspection and structure

### 3. Preprocessing
- Label encoding of categorical features
- Standard scaling of features for SVM and Logistic Regression pipelines with balanced data

### 4. Model Training
- Train/test split with stratification
- Pipelines are used for SVM and Logistic Regression to combine scaling + model
- All models are tuned with RandomizedSearchCV for optimal hyperparameters

> All models are now encapsulated in pipelines for robust deployment.

### 5. Evaluation
- Accuracy, precision, recall, F1-score, and AUC-ROC
- Confusion matrices for misclassifications
- ROC curves for model comparison
- Random Forest feature importance (top 10 features)
- Boxplot: call duration by subscription result

## File Structure
```
.
├── data/
├── models/
├── .gitignore
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

## Individual Modules
**main.py**
The main script, it runs everything: loads data, trains models, evaluates, visualizes.

**modeling.py**
Machine learning core: creates pipelines, tunes hyperparameters, trains Random Forest/SVM/Logistic Regression.

**visualization.py**
Plotting module: creates age distributions, confusion matrices, ROC curves, feature importance, duration analysis.

**utils.py**
Utilities: saves/loads trained models as .pkl files.

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
