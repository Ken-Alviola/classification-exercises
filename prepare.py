import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prep_iris(df):
    '''
    Takes a iris dataframe drops species_id and measurement_id,renames species_name to just species and creates dummy variables for species
    return: single cleaned dataframe
    '''
    df.drop_duplicates(inplace=True)
    dropcols = ['species_id','measurement_id']
    df.drop(columns=dropcols, inplace=True)
    df.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    dummies = pd.get_dummies(df[['species']], drop_first=False)
    return pd.concat([df, dummies], axis=1)