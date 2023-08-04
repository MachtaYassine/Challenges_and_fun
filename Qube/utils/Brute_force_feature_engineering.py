#%% 

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

def get_sign(x):
    return x>0

def log_column(x):
    if get_sign(x):
        return np.log10(x)
    else:
        return -np.log10(-x)
    
def root_column(x,i):
    if get_sign(x):
        return x**(1/i)
    else:
        return -((-x)**(1/i)) 


def add_columns_powered(df,base_columns, upperbound, invert=False):
    for column in base_columns :
        for i in range(2, upperbound+1):
            df[column + '_power' + str(i)] = df[column] ** i
            if invert:
                df[column + '_root'+str(i)] = df[column].apply(root_column,args=(i,))
        # df[column + '_log'] = df[column].apply(log_column)
        


def get_subparts(base_columns):
    if len(base_columns) == 0:
        return [[]]
    
    first_elem = base_columns[0]
    remaining_subparts = get_subparts(base_columns[1:])
    subparts = [subset + [first_elem] for subset in remaining_subparts] + remaining_subparts
    return subparts

def get_subparts2(df, base_columns):
    subparts = [[]]

    for first_elem in base_columns:
        new_subparts = []
        for subset in subparts:
            new_subset = subset + [first_elem]
            new_subparts.append(new_subset)
        subparts += new_subparts

    return subparts


def correlation_sort(df,target,thresh):
    correlations = df.corrwith(target['TARGET'])
    filtered_correlations = correlations[abs(correlations) > thresh]
    return filtered_correlations.sort_values(ascending=False)


def add_n_uplet(df,base_columns,n):
    subparts=get_subparts(base_columns)
    filtered_list = [ element for element in subparts if len(element) <= n and len(element) > 1 ]
    length=len(filtered_list)
    print(length , "colunms to add")
    
    for column_list in tqdm(filtered_list):
        column_title= ''
        for title in column_list:
            column_title+="_x_"+title
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[column_title]= df[column_list].product(axis=1)
        
    
def main_brute(df,base_columns,target,thresh=0.2,upperbound=3,n_uplet=6,invert=True):
    df_copy=df.copy(deep=True)
    add_columns_powered(df_copy,base_columns,upperbound,invert)
    print("powers added")
    add_n_uplet(df_copy,base_columns,n_uplet)
    print("n_uplet added")
    corr=correlation_sort(df_copy,target,thresh)
    return df_copy, corr
#%% 
# df = pd.read_csv('X_dropped_DE.csv').drop(['ID',"DAY_ID","DE_RAIN","DE_WIND","DE_TEMP"], axis=1)
# base_columns=list(df.columns)
# # print(base_columns)
# target=pd.read_csv('Y_dropped_DE.csv').drop('ID', axis=1)
# # add_columns_powered(df,base_columns,3,True)

# _,corr=main_brute(df,base_columns,target,n_uplet=10)

# corr

