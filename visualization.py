import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

class Visualizer:

    def __init__(self, figsize=(8, 5), fontweight="bold"):
        self.figsize = figsize
        self.fontweight = fontweight

    def age_distribution(self, df):
        plt.figure(figsize=self.figsize)

        sns.histplot(
            data=df,
            x="age",
            hue="accepts",
            multiple="stack",
            bins=20,
            palette="viridis"
        )

        plt.xticks(range(int(df["age"].min()), int(df["age"].max()) + 1, 5))

        plt.title("Age Distribution by Subscription", fontweight=self.fontweight)
        plt.tight_layout()
        plt.show()

    def roc_comparison(self, y_test, rf_proba, svm_proba, lr_proba):
        _, ax = plt.subplots(figsize=self.figsize)

        RocCurveDisplay.from_predictions(
            y_test,
            svm_proba,
            name="SVM",
            ax=ax
        )

        RocCurveDisplay.from_predictions(
            y_test,
            rf_proba,
            name="Random Forest",
            ax=ax
        )

        RocCurveDisplay.from_predictions(
            y_test,
            lr_proba,
            name="Logistic Regression",
            ax=ax
        )

        ax.set_title("ROC Curve Comparison", fontweight=self.fontweight)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        plt.show()
