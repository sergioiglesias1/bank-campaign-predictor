# Bank Marketing Campaign Predictions

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

### This project applies machine learning techniques to predict whether a client will subscribe to a bank term deposit after a marketing call, using the UCI Bank Marketing Dataset. The focus here is on identifying potential subscribers (maximizing recall) while keeping operational costs low.

## Project Overview
This repository compares different machine learning models to classify client responses (deposit vs no deposit).  
The models include:

- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **Logistic Regression**

> All with hyperparameter tuning via RandomizedSearchCV

### Why these models?
- **Logistic Regression**: It is a strong base for binary classifications and allows us to understand the impact of each feature on the probability of subscription.
- **Support Vector Machine (SVM)**: It attempts to find the optimal hyperplane that maximizes the margin between the two classes. Optimal for classification.
- **Random Forest**: It is a model with less likely to overfitting than individual decision trees. It also provides valuable data for feature importance.

## Pipeline Overview

- **Data**: Load dataset, encode target (`yes` → 1, `no` → 0), train/test split with stratification  
- **Preprocessing & Architecture**: OOP classes (`ModelTrainer`, `Visualizer`, `ModelSaver`); numerical features scaled, categorical one-hot encoded  
- **Modeling**: Pipelines for SVM, Logistic Regression, Random Forest; hyperparameter tuning with `RandomizedSearchCV` (ROC-AUC)  
- **Evaluation**: Metrics (Accuracy, Precision, Recall, F1, AUC-ROC) and plots (Confusion Matrix, ROC, Feature Importance, call duration analysis)

## Data Sources
- UCI Bank Marketing Dataset (`bank-additional-full.csv`)
- Link to data: [Kaggle Bank Marketing Dataset](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv/data)
- Contains client demographic data, call duration, campaign info, and economic indicators

All dataset files are stored in the `data/` folder.

## How It Works
- Run `EDA.ipynb` first to analyze the data, detect outliers, and generate `data/cleaned_data.csv`.
- Run `main.py` which loads the cleaned data.
- The script applies scaling/encoding via pipelines and trains SVM, Logistic Regression, and Random Forest.
- The main file runs all the python files with classes and functions to train the models and save them in the `models/` folder.

> All models are now encapsulated in pipelines for robust deployment.

### 5. Evaluation
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

> SVM prioritizes recall on the positive class, making it more suitable for marketing campaigns where the cost of missing a potential subscriber is higher than the cost of an extra call.

> Although SVM achieves higher recall, **Logistic Regression provides a better precision–recall tradeoff**, resulting in fewer wasted calls per captured subscriber.

---

### Notes on Model Interpretation

- **High Accuracy vs Low Recall**: Even with `class_weight='balanced'` and stratified splits, Random Forest achieves high overall accuracy (~90%) but low recall on subscribed clients. This is due to the strong class imbalance (~11% positives) and the default decision threshold, which still favors the majority class. Accuracy remains dominated by correct negative predictions, while minority-class recall requires threshold or objective-level optimization.

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
> **The simplest model (Logistic Regression) delivers the best business-aligned performance**, even though SVM has a better recall, Logistic Regression has a lot more precision and F1, so it captures more positive cases while maintaining a better balance between missed positives and false alarms.

## File Structure
```
.
├── data/
├── models/
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

## Key Files
- **EDA.ipynb**
Jupyter Notebook for Exploratory Data Analysis. It visualizes distributions, handles outliers (IQR method), and exports the `cleaned_data.csv` used for modeling.

- **main.py**
The main script, it runs everything: loads data, trains models, evaluates, visualizes.

- **modeling.py**
Machine learning core: creates pipelines, tunes hyperparameters, trains Random Forest/SVM/Logistic Regression.

- **visualization.py**
Plotting module: creates age distributions, confusion matrices, ROC curves, feature importance, duration analysis.

- **utils.py**
Utilities: saves/loads trained models as .pkl files.

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
