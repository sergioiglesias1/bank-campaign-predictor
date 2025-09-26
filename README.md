# Predictions on a Bank Campaign

This project applies machine learning techniques to predict whether a client will subscribe to a bank term deposit after a marketing call, using the **UCI Bank Marketing Dataset**.  

---

## Project Overview  
This repository compares different machine learning models to classify client responses (`deposit` vs `no deposit`).  
The models include:  

- **Support Vector Machine (SVM)** with RBF kernel  
- **Random Forest Classifier**  
- **Logistic Regression**  

The pipeline involves:  
- Preprocessing categorical variables (label encoding + one-hot encoding)  
- Data scaling  
- Training multiple classifiers  
- Evaluating performance with accuracy, precision, recall, F1-score, and AUC-ROC  
- Visualizing results with confusion matrices, ROC curves, and feature importance  

---

## Data Sources  
- **UCI Bank Marketing Dataset** (`bank-additional-full.csv`)  
  - Link to data here: https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv/data
  - Contains client demographic data, call duration, campaign info, and economic indicators
---

## How It Works  
1. **Data Loading & Cleaning**  
   - Load CSV, sample 3-5% of data for faster training  
   - Encode target variable (`yes/no → 1/0`)  

2. **Exploratory Analysis**  
   - Age vs subscription (swarmplot)  
   - Call duration distribution  

3. **Preprocessing**  
   - One-hot encoding of categorical features  
   - Standard scaling of features  

4. **Model Training**  
   - Train/test split with stratification  
   - Fit Logistic Regression, Random Forest, and SVM  

5. **Evaluation**  
   - Accuracy, precision, recall, F1-score, AUC-ROC  
   - Confusion matrices and ROC curves for comparison  

---

## Dependencies  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

---

## File Structure  
- `prediction1.py` → Main training + evaluation pipeline  
- `data/` → Dataset (not included, must be downloaded separately from UCI repository)  
- `plots/` → Generated plots (confusion matrix, ROC curves, feature importance, boxplots)  

---

## Usage  
Run the script with:  

```bash
python3 bank_ml.py
```

Expected output:  

```
======================================================================
PERFORMANCE METRICS PER MODEL
======================================================================
Model           Accuracy   Precision   Recall     F1-Score    AUC-ROC
----------------------------------------------------------------------
SVM             0.914      0.718       0.385      0.501       0.942
Random Forest   0.916      0.692       0.457      0.550       0.949
LogReg          0.865      0.452       0.912      0.604       0.944
======================================================================

```  

---

## Model Performance  
- **Metrics used:** Accuracy, Precision, Recall, F1-score, AUC-ROC  
- **Visualization:**  
  - Confusion matrix (misclassifications)  
  - ROC curve comparison across models  
  - Random Forest feature importance (top 10 features)  
  - Boxplot: call duration by subscription result  

---

## Future Improvements  
- Add **hyperparameter tuning** with GridSearchCV/RandomizedSearchCV  
- Use **ensemble stacking** to combine models  
- Test **Gradient Boosting** and **XGBoost**  
- Incorporate external macroeconomic data for richer features  
- Deploy a simple web app for real-time predictions  

---

## License  
This project is licensed under the MIT License.  
