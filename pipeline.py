
# Preprocessing
from imblearn.pipeline import Pipeline


def get_pipline(estimator, data):
    
    preprocessor = get_column_tranformer(data)
    # pipline building
    return Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classification', estimator)
    ])


def get_column_tranformer(data):
    #remove as a first step all NA later on we can try to impute 
    x = data.dropna()

    # numerical values
    number_cols = data.select_dtypes(include='number')

    #categorical values
    sector_mapping = {
    'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4,
    'G5': 5, 'G6': 6, 'G7': 7, 'G8': 8,
    'G9': 9, 'G10': 10, 'G11': 11
    }   

    data['Group'] = data['Group'].map(sector_mapping)

    #remove result columns do before pipeline
    #data = data.drop(columns=['Class','Perform'])