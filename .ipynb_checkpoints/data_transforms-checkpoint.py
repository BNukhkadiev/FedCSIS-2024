from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



def xy_split(data):
    # splits data into features and target
    X = data.drop(columns=['Class','Perform'])  
    y = data['Class']
    return X, y

def encode(data):
    # label encodes target to be 0, 1, 2
    encoder = LabelEncoder()
    tdata = data.copy()
    tdata['Class'] = tdata[['Class']].apply(encoder.fit_transform)
    return {
        'enc': encoder,
        'data': tdata, 
           }

def decode(encoder, data):
    # returns python list decoding back to -1, 0, 1
    return encoder.inverse_transform(data.ravel()).tolist()

def dummies(data):
    # creates dummies from Group col
    d = data.copy()
    d = pd.concat((d.drop(columns=['Group']), pd.get_dummies(d['Group'])), axis=1)
    return d

def submit(path, predictions):
    # writes submition file
    with open(path, 'w') as f:
        for prediction in predictions.tolist():
            f.write(f"{prediction}\n")  # Writing each prediction on a new line
    
    print("Submission file created:", path)
    return True

