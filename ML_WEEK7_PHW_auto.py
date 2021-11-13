
import pandas as pd
from sklearn import preprocessing


#onehot Encoding
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def getEncode(df,name,encoder):
    encoder.fit(df[name])
    labels = encoder.transform(df[name])
    df.loc[:, name] = labels

def onehotEncode(df, name):
   le = preprocessing.OneHotEncoder(handle_unknown='ignore')
   enc = df[[name]]
   enc = le.fit_transform(enc).toarray()
   enc_df = pd.DataFrame(enc, columns=le.categories_[0])
   df.loc[:, le.categories_[0]] = enc_df
   df.drop(columns=[name], inplace=True)

#label encoding
def labelEncode(df, name):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df[name])
    labels = encoder.transform(df[name])
    df.loc[:, name] = labels

"""
:param X: feature values
:param numerical_columns: name of numerical columns (array of string)
:param categorical_columns: name of categorical columns (array of string)
:param scalers: array of scalers
:param encoders: array of encoders 
:param scaler_name: name of scalers (array of string)
:param encoder_name: name of encoders (array of string)
:return: 2d array that is scaled and encoded X 
"""
def get_various_encode_scale(X, numerical_columns, categorical_columns, scalers=None, encoders= None,scaler_name=None,encoder_name=None):

    if categorical_columns is None:
        categorical_columns = []
    if numerical_columns is None:
        numerical_columns = []
    if len(categorical_columns) == 0:
        return get_various_scale(X,numerical_columns,scalers,scaler_name)
    if len(numerical_columns) == 0:
        return get_various_encode(X,categorical_columns,encoders,encoder_name)

    """
    Test scale/encoder sets
    """
    if scalers is None:
        scalers = [preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.RobustScaler()]
    if encoders is None:
        encoders = [preprocessing.LabelEncoder(),preprocessing.OneHotEncoder()]

    after_scale_encode = [[0 for col in range(len(encoders))] for row in range(len(scalers))]

    i=0
    for scale in scalers:
        for encode in encoders:
            after_scale_encode[i].pop()
        for encode in encoders:
            after_scale_encode[i].append(X.copy())
        i=i+1

    for name in numerical_columns:
        i=0
        for scaler in scalers:
            j=0
            for encoder in encoders:
                after_scale_encode[i][j][name] = scaler.fit_transform(X[name].values.reshape(-1, 1))
                j=j+1
            i=i+1

    for new in categorical_columns:
        i=0
        for scaler in scalers:
            j=0
            for encoder in encoders:
                if (str(type(encoder)) == "<class 'sklearn.preprocessing._label.LabelEncoder'>"):
                    labelEncode(after_scale_encode[i][j], new)
                elif (str(type(encoder)) == "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>"):
                    onehotEncode(after_scale_encode[i][j], new)
                else:
                    getEncode(after_scale_encode[i][j], new, encoder)
                j=j+1
            i=i+1

    return after_scale_encode,scalers,encoders

"""
If there aren't categorical value, do this function
This function only scales given X
Return: 1d array of scaled X
"""
def get_various_scale(X, numerical_columns, scalers=None,scaler_name=None):

    """
    Test scale/encoder sets
    """
    if scalers is None:
        scalers = [preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.RobustScaler()]
        #scalers = [preprocessing.StandardScaler()]
    encoders = ["None"]

    after_scale = [[0 for col in range(1)] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_scale[i].pop()
        for encode in encoders:
            after_scale[i].append(X.copy())
        i = i + 1

    for name in numerical_columns:
       i=0
       for scaler in scalers:
           after_scale[i][0][name] = scaler.fit_transform(X[name].values.reshape(-1,1))
           i=i+1

    return after_scale,scalers,["None"]

"""
If there aren't numerical value, do this function
This function only encodes given X
Return: 1d array of encoded X
"""
def get_various_encode(X, categorical_columns, encoders=None,encoder_name=None):

    """
    Test scale/encoder sets
    """
    if encoders is None:
        #encoders = [preprocessing.LabelEncoder(),preprocessing.OneHotEncoder()]
        encoders = [preprocessing.LabelEncoder()]
    scalers = ["None"]

    after_encode = [[0 for col in range(1)] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_encode[i].pop()
        for encode in encoders:
            after_encode[i].append(X.copy())
        i = i + 1

    for new in categorical_columns:
        j = 0
        for encoder in encoders:
            if (str(type(encoder)) == "<class 'sklearn.preprocessing._label.LabelEncoder'>"):
                labelEncode(after_encode[0][j], new)
            elif (str(type(encoder)) == "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>"):
                onehotEncode(after_encode[0][j], new)
            else:
                getEncode(after_encode[0][j], new, encoder)
            j = j + 1


    return after_encode,["None"],encoders


def get_result(data):
    # 1. job
    # 2. martial
    # 3. education
    # 5. housing loan
    # 6. personal loan
    # 7. contact communication type
    # 14. outcome of the previous marketing campaign
    # 19. has the client subscribed a term deposit?
    X = data.iloc[:,0:20]
    y = data.loc[:,'y']
    numerical_columns = ['age','duration','campaign', 'pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
    categorical_columns = ['job','marital','education','default','housing','loan','contact','month','day_of_week','duration','poutcome']
    X_preprocessed,scalers, encoders = get_various_encode_scale(X,numerical_columns,categorical_columns)
    model = GaussianNB()

    i=0
    for scaler in scalers:
        j=0
        for encoder in encoders:
            data = X_preprocessed[i][j]
            train_x, test_x, train_y, test_y = train_test_split(data,y)
            model.fit(train_x,train_y)
            print("Score with ",scaler," & ",encoder," combination: ",model.score(test_x,test_y))
            j = j+1
        i=i+1


