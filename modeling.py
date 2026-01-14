from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def __init__(
        self, 
        random_state=42
    ):

        self.random_state = random_state
        self.pipelines = {}
        self.param_grids = {}

    def preprocessor(self, df):
        # categorical & numerical columns
        self.cat_cols = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome"
        ]

        self.num_cols = [x for x in df.columns if x not in self.cat_cols + ["accepts"]]
        
        return ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.cat_cols)
            ]
        )

    def make_pipelines(self, df):
        preprocessor = self.preprocessor(df)
        self.pipelines = {
            'rf': Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestClassifier(class_weight='balanced', random_state=self.random_state))
            ]),
            'svm': Pipeline([
                ('preprocessor', preprocessor),
                ('model', SVC(probability=True, class_weight='balanced', random_state=self.random_state))
            ]),
            'logreg': Pipeline([
                ('preprocessor', preprocessor),
                ('model', LogisticRegression(max_iter=1000, class_weight='balanced', fit_intercept=True, random_state=self.random_state))
            ])
        }

    def model_params(self):
        self.param_grids = {
            'svm': {'model__C': [0.1, 1, 10], 'model__gamma': [0.01, 0.1, 1]},
            'logreg': {'model__C': [0.01, 0.1, 1, 10]},
            'rf': {
                'model__n_estimators': [100, 300], 
                'model__max_depth': [5, 10, 20],
                'model__min_samples_leaf': [1, 3]
            }
        }

    def hyperparameter_search(
        self,
        X_train,
        y_train
    ):
        results = {}

        for name, pipe in self.pipelines.items():
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=self.param_grids[name],
                cv=3,
                n_iter=10,
                scoring="roc_auc",
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )

            search.fit(X_train, y_train)

            results[name] = {
                "best_estimator": search.best_estimator_,
                "best_score": search.best_score_,
                "best_params": search.best_params_
            }

            print(
                f"Best AUC for {name}: "
                f"{search.best_score_:.3f} "
                f"with params {search.best_params_}"
            )

        return results
