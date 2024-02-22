from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, make_scorer
 
 
class Classifier(BaseEstimator):
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = ExtraTreesClassifier(
            random_state=42,
            class_weight='balanced_subsample'
        )
        #self.smote = SMOTE(k_neighbors=1)
        self.under_sampler = RandomUnderSampler(random_state=42)
        self.pipeline = make_pipeline_imb(
            self.imputer,
            self.scaler,
            #self.smote,
            self.under_sampler,
            self.model
        )
        self.features = [ 'K2O_normalized', 'SiO2_normalized', 'MgO_normalized', 'TiO2_normalized', 'CaO_normalized', 'Zr', 'Nb', 'Rb', 'Y', 'Cs', 'Th', 'Sr', 'Ce', 'Yb', 'Pb' ]# 用于存储选择的特征
 
    def fit(self, X, y):
        X = X[self.features]

        param_grid = {
            'extratreesclassifier__n_estimators': [300],
            'extratreesclassifier__max_depth': [20],
            'extratreesclassifier__min_samples_split': [4],
            'extratreesclassifier__min_samples_leaf': [4],
            'extratreesclassifier__max_features': ['sqrt', 'log2'],
        }
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid=param_grid,
            scoring=make_scorer(balanced_accuracy_score),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=-1
        )
        grid_search.fit(X, y)
        self.best_estimator_ = grid_search.best_estimator_
 
    def predict(self, X):
        return self.best_estimator_.predict(X)
 
    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)
