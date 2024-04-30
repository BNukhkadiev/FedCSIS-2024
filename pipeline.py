
# Preprocessing
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import LinearSVC
from sklearn.compose import make_column_transformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer


def get_pipline(estimator, data):
    
    preprocessor = get_column_tranformer(data)
    # pipline building
    return Pipeline(steps=[
        
        ('preprocessing', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k=2)),
        ('classification', estimator)
    ])


def get_column_tranformer(data):

    # numerical values
    numerical_cols = data.select_dtypes(include='number').columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])
    #categorical values
    group_transfromer = GroupTransformer()
    #remove result columns do before pipeline
    return make_column_transformer(
        (numerical_transformer, numerical_cols),
        (group_transfromer,['Group'])
        )

class GroupTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        sector_mapping = {
        'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4,
        'G5': 5, 'G6': 6, 'G7': 7, 'G8': 8,
        'G9': 9, 'G10': 10, 'G11': 11
        }   

        X_copy['Group'] = X_copy['Group'].map(sector_mapping)
        return X_copy