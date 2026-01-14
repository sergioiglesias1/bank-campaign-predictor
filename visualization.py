import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

class Visualizer:

    def __init__(self):
        self.figsize = (8, 5)
        self.fontweight = "bold"

    def age_distribution(
        self,
        df
    ):
        plt.figure(figsize=self.figsize)
        sns.histplot(
            data=df,
            x="age",
            hue="accepts",
            common_norm=False,
            kde=True,
            fill=True,
            alpha=0.3
        )
        plt.xticks(range(int(df["age"].min()), int(df["age"].max()) + 1, 5))
        plt.title("Age Distribution by Subscription")
        plt.tight_layout()
        plt.show()

    def confusion_matrix_lr(
        self,
        y_test,
        y_pred_lr
    ):
        cm = confusion_matrix(y_test, y_pred_lr)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No", "Yes"]
        )

        disp.plot(
            cmap="viridis"
        )

        plt.title("Logistic Regression Confusion Matrix", fontweight=self.fontweight)
        plt.show()

        tn, fp, fn, tp = cm.ravel()

        print(f"Wasted calls (Type I Error): {fp}")
        print(f"Lost clients (Type II Error): {fn}")
        print(f"Predicted Acceptance Rate: {(y_pred_lr.sum()/len(y_test))*100:.1f}%")
        print(f"Real Acceptance Rate: {(y_test.mean()*100):.1f}%")

    def roc_comparison(
            self, 
            y_test,
            y_proba_rf, 
            y_proba_svm, 
            y_proba_lr
    ):
        _, ax = plt.subplots(figsize=self.figsize)

        RocCurveDisplay.from_predictions(y_test, y_proba_svm, name="SVM", ax=ax)
        RocCurveDisplay.from_predictions(y_test, y_proba_rf, name="Random Forest", ax=ax)
        RocCurveDisplay.from_predictions(y_test, y_proba_lr, name="Logistic Regression", ax=ax)

        ax.set_title("ROC Curve Comparison", fontweight=self.fontweight)
        ax.legend()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

    def feature_importance(
        self,
        best_rf
    ):
        try:
            rf_model = best_rf.named_steps["model"]
            preprocessor = best_rf.named_steps["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()

            if len(feature_names) != len(rf_model.feature_importances_):
                raise ValueError(
                    f"Feature mismatch: {len(feature_names)} features vs "
                    f"{len(rf_model.feature_importances_)} importances"
                )

            feature_importance = (
                pd.DataFrame({
                    "feature": feature_names,
                    "importance": rf_model.feature_importances_
                })
                .sort_values("importance", ascending=False)
                .head(10)
            )

            plt.figure(figsize=(8, 5))
            sns.barplot(
                y="feature",
                data=feature_importance,
                palette="viridis",
                x="importance"
            )
            plt.title("Top 10 - Feature Importance", fontweight=self.fontweight)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"[Feature Importance Plot Error] {e}")

    def call_duration_boxplot(
        self,
        df
    ):
        df_plot = df.copy()
        df_plot['Subscription'] = df_plot['accepts'].map({0: 'No', 1: 'Yes'})

        plt.figure(figsize=self.figsize)
        sns.boxplot(
            x='Subscription',
            y='duration',
            data=df_plot,
            palette='Set1'
        )
        plt.title('Call Duration by Subscription', fontweight=self.fontweight)
        plt.tight_layout()
        plt.show()