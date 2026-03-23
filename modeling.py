from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
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
        num_cols = X_train.select_dtypes(include=['np.number']).columns
        cat_cols = X_train.select_dtypes(include=['object']).columns
        
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('to_str', FunctionTransformer(lambda x: x.astype(str))),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])

        return ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ]
        )
        
    def make_pipelines(self, X_train):
        self.pipelines = {
            'rf': Pipeline([
                ('preprocessor', self.make_preprocessor(X_train)),
                ('model', RandomForestClassifier(class_weight='balanced', random_state=self.random_state))
            ]),
            'svm': Pipeline([
                ('preprocessor', self.make_preprocessor(X_train)),
                ('model', SVC(probability=True, class_weight='balanced', random_state=self.random_state))
            ]),
            'logreg': Pipeline([
                ('preprocessor', self.make_preprocessor(X_train)),
                ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=self.random_state))
            ])
        }
    
    def model_params(self):
        self.param_grids = {
            'svm': [
                {
                    'model__kernel': ['rbf'],
                    'model__C': [0.1, 1, 10],
                    'model__gamma': ['scale', 0.01, 0.1],
                },
                {
                    'model__kernel': ['linear'],
                    'model__C': [0.1, 1, 10],
                },
            ],
            'logreg': {
                'model__C': [0.01, 0.1, 1, 10],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['liblinear', 'saga']
            },
            'rf': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [5, 10, 15, 20],
                'model__min_samples_leaf': [1, 2]
            }
        }
        
    def hyperparameter_search(self, X_train, y_train):
        stkfold = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )
        
        results = {}
        
        for name, pipe in self.pipelines.items():
            search = GridSearchCV(
                estimator=pipe,
                param_grid=self.param_grids[name],
                cv=stkfold,
                scoring="roc_auc",
                n_jobs=-1,
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
