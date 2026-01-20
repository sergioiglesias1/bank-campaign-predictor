from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipelines = {}
        self.param_grids = {}

    def make_preprocessor(self, X_train):
        num_cols = ['age', 'campaign', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'nr.employed']
        cat_cols = [x for x in X_train.columns if x not in num_cols]
        
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        return ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ]
        )

    def make_pipelines(self, X_train):
        preprocessor = self.make_preprocessor(X_train)
        
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
            'svm': {
                'model__C': [0.1, 1, 10],
                'model__gamma': [0.01, 0.1, 1]
            },
            'logreg': { 
                'model__C': [0.01, 0.1, 1, 10]
            },
            'rf': {
                'model__n_estimators': [100, 300], 
                'model__max_depth': [5, 10, 20],
                'model__min_samples_leaf': [1, 3]
            }
        }

    def hyperparameter_search(self, X_train, y_train):
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
            
            print(f"Best AUC for {name}: {search.best_score_:.3f} with params {search.best_params_}")
            
        return results
