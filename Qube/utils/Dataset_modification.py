import pandas as pd
import warnings


datasets="/home/ymachta/Challenges_and_fun/Qube/Datasets/"

def seperate_data_by_countries(X_train,Y_train,drop_or_fill_na="fill",embedded_countries=True,rank=True):
    with warnings.catch_warnings():
        if embedded_countries:
            country_mapping = {'FR': 0, 'DE': 1}
            X_train['COUNTRY'] = X_train['COUNTRY'].map(country_mapping)
            # Create sub-dataset 'trainX_DE' with rows where ID is 'DE' and keep columns starting with 'ID'
            X_train_DE = X_train[X_train['COUNTRY']==1].loc[:, X_train.columns.str.contains('^(ID|DAY_ID|DE|GAS|COAL|CARBON)')]
            # Create sub-dataset 'trainX_FR' with rows where ID is 'FR' and keep columns starting with 'ID'
            X_train_FR = X_train[X_train['COUNTRY']==0].loc[:, X_train.columns.str.contains('^(ID|DAY_ID|FR|GAS|COAL|CARBON)')]
        else:
            # Create sub-dataset 'trainX_DE' with rows where ID is 'DE' and keep columns starting with 'ID'
            X_train_DE = X_train[X_train['COUNTRY'].str.startswith('DE')].loc[:, X_train.columns.str.contains('^(ID|DAY_ID|DE|GAS|COAL|CARBON)')]
            # Create sub-dataset 'trainX_FR' with rows where ID is 'FR' and keep columns starting with 'ID'
            X_train_FR = X_train[X_train['COUNTRY'].str.startswith('FR')].loc[:, X_train.columns.str.contains('^(ID|DAY_ID|FR|GAS|COAL|CARBON)')]
    X_train_FR=X_train_FR.drop("FR_NET_IMPORT",axis=1)
    X_train_DE=X_train_DE.drop("DE_NET_IMPORT",axis=1)
    Y_train_DE = Y_train.loc[X_train_DE.index]
    Y_train_FR = Y_train.loc[X_train_FR.index]
    if rank:
        Y_train_FR['Rank']= Y_train_FR['TARGET'].rank().astype(int)
        Y_train_DE['Rank']= Y_train_DE['TARGET'].rank().astype(int)
        Y_train_FR['Rank_group']=pd.qcut(Y_train_FR.Rank,2).cat.codes
        Y_train_DE['Rank_group']=pd.qcut(Y_train_DE.Rank,2).cat.codes

    
    X_modified_DE,Y_modified_DE = handle_na(X_train_DE,Y_train_DE,drop_or_fill_na)
    X_modified_FR,Y_modified_FR = handle_na(X_train_FR,Y_train_FR,drop_or_fill_na)

    return X_modified_DE,Y_modified_DE,X_modified_FR,Y_modified_FR
   
    

def handle_na(X_train,Y_train,drop_or_fill_na="fill"):
    merged=pd.merge(X_train,Y_train, on='ID')
    Data = merged.copy(deep=True)
    # Fill missing values with the mean (you can choose other imputation strategies as well)
    if drop_or_fill_na=='fill':
        Data.fillna(Data.mean(), inplace=True)
    elif drop_or_fill_na=='drop': 
        Data.dropna(inplace=True)
    # Split the dataset into features (X) and target (Y)
    X_modified = Data.drop(columns=['ID','DAY_ID','TARGET','Rank','Rank_group'])
    Y_modified = Data[['TARGET','Rank','Rank_group']]
    return X_modified,Y_modified