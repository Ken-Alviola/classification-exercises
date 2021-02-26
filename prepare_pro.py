import pandas as pd
from sklearn.model_selection import train_test_split

def clean_iris(df):
    '''
    clean_iris will take one argument df, a pandas dataframe, anticipated to be the iris dataset
    and will remove species_id and measurement_id columns,
    rename species_name to species
    encode species into two new columns
    
    return: a single pandas dataframe with the above operations performed
    '''
    df = df.drop(['species_id', 'measurement_id'], axis=1)
    df.rename(columns={'species_name': 'species'}, inplace=True)
    dummies = pd.get_dummies(df[['species']], drop_first=False)
    return pd.concat([df,dummies], axis=1)


def prep_iris(df):
    '''
    prep_iris will take one argument df, a pandas dataframe, anticipated to be the iris dataset
    and will remove species_id and measurement_id columns,
    rename species_name to species
    encode species into two new columns
    
    perform a train, validate, test split
    
    return: three pandas dataframes: train, validate, test
    '''
    
    df = clean_iris(df)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=1349, stratify=df.species)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=1349, stratify=train_validate.species)
    return train, validate, test

def generic_split(df, stratify_by=None):
    """
    Crude train, validate, test split
    To stratify, send in a column name
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.3, random_state=123)
        train, validate = train_test_split(df, test_size=.3, random_state=123)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(df, test_size=.3, random_state=123, stratify=train[stratify_by])
    
    return train, validate, test