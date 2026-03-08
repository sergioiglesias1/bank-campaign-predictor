import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

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

    def confusion_matrix_lr(self, y_test, y_pred_lr):
        cm = confusion_matrix(y_test, y_pred_lr)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No", "Yes"]
        )

        disp.plot(cmap="viridis")

        plt.title(
            "Logistic Regression Confusion Matrix",
            fontweight=self.fontweight
        )

        plt.show()

        tn, fp, fn, tp = cm.ravel()

        print(f"Wasted calls (Type I Error): {fp}")
        print(f"Lost clients (Type II Error): {fn}")
        print(f"Predicted Acceptance Rate: {(y_pred_lr.sum()/len(y_test))*100:.1f}%")
        print(f"Real Acceptance Rate: {(y_test.mean()*100):.1f}%")

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