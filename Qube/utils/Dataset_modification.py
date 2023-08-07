import pandas as pd
import warnings
import numpy as np
import itertools

datasets="/home/ymachta/Challenges_and_fun/Qube/Datasets/"

def seperate_data_by_countries(X_train,Y_train,drop_or_fill_na="fill",embedded_countries=True,rank_before_split=False,custom_group=2,keep_id=False):
    if rank_before_split:
        Y_train['Rank']= Y_train['TARGET'].rank().astype(int)
        Y_train['Rank_group']=pd.qcut(Y_train.Rank,custom_group).cat.codes
        

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    if not rank_before_split:
        Y_train_FR['Rank']= Y_train_FR['TARGET'].rank().astype(int)
        Y_train_DE['Rank']= Y_train_DE['TARGET'].rank().astype(int)
        Y_train_FR['Rank_group']=pd.qcut(Y_train_FR.Rank,2).cat.codes
        Y_train_DE['Rank_group']=pd.qcut(Y_train_DE.Rank,2).cat.codes

    
    X_modified_DE,Y_modified_DE = handle_na(X_train_DE,Y_train_DE,drop_or_fill_na,keep_id)
    X_modified_FR,Y_modified_FR = handle_na(X_train_FR,Y_train_FR,drop_or_fill_na,keep_id)

    return X_modified_DE,Y_modified_DE,X_modified_FR,Y_modified_FR
   
    

def handle_na(X_train,Y_train,drop_or_fill_na,keep_id):
    merged=pd.merge(X_train,Y_train, on='ID')
    Data = merged.copy(deep=True)
    # Fill missing values with the mean (you can choose other imputation strategies as well)
    if drop_or_fill_na=='fill':
        Data.fillna(Data.mean(), inplace=True)
    elif drop_or_fill_na=='drop': 
        Data.dropna(inplace=True)
    # Split the dataset into features (X) and target (Y)
    if not keep_id:
        X_modified = Data.drop(columns=['ID','DAY_ID','TARGET','Rank','Rank_group'])
        Y_modified = Data[['TARGET','Rank','Rank_group']]
    else:
        X_modified = Data.drop(columns=['DAY_ID','TARGET','Rank','Rank_group'])
        Y_modified = Data[['ID','TARGET','Rank','Rank_group']]
    return X_modified,Y_modified

def create_pairwise_dataset(df):
    """
    Create a pairwise dataset for comparing IDs.

    Args:
        df: A pandas DataFrame with a column 'IDs'.

    Returns:
        A DataFrame containing pairwise comparisons.
    """
    # Get unique IDs from the DataFrame

    # Generate all possible pairs of unique IDs
    pairs = list(itertools.combinations(df[['ID','TARGET']].values, 2))
    # print("pairs",pairs)
    # Create a list to store pairwise comparisons
    pairwise_data = []

    # Compare each pair of IDs and add to the list
    for (id1,target1), (id2,target2) in pairs:
        comparison = 1.0 if target1 > target2 else (0.5 if target1 == target2 else 0.0)
        pairwise_data.append((id1, id2, comparison))

    # Create a DataFrame from the pairwise data
    pairwise_df = pd.DataFrame(pairwise_data, columns=['ID1', 'ID2', 'Comparison'])

    return pairwise_df



def reconstruct_original_dataset(pairwise_df):
    """
    Reconstruct the original dataset from a pairwise dataset.

    Args:
        pairwise_df: A pandas DataFrame with columns 'ID1', 'ID2', and 'Comparison'.

    Returns:
        A pandas DataFrame with columns 'IDs' and 'Target'.
    """
    # Get unique IDs from the pairwise DataFrame
    unique_ids = set(pairwise_df['ID1']).union(pairwise_df['ID2'])

    # Create a dictionary to map IDs to their corresponding targets
    id_to_target = {}
    dict_of_winning_comparaison = {}

    # Iterate through rows in the pairwise DataFrame to determine targets for each ID
    for _, row in pairwise_df.iterrows():
        if row['Comparison'] > 0.5:
            # print('row',row)
            if row['ID1'] in id_to_target.keys() and row['ID1'] in dict_of_winning_comparaison.keys():
                id_to_target[row['ID1']] +=1 
                dict_of_winning_comparaison[row['ID1']] +=[row['ID2']] 
            else: 
                id_to_target[row['ID1']] =2
                dict_of_winning_comparaison[row['ID1']] =[row['ID2']] 
                
            if not row['ID2'] in id_to_target.keys(): id_to_target[row['ID2']] =1
        elif row['Comparison'] <0.5:
            # print('row',row)
            if not row['ID1'] in id_to_target.keys(): id_to_target[row['ID1']] =1
            if row['ID2'] in id_to_target.keys() and row['ID2'] in dict_of_winning_comparaison.keys(): 
                id_to_target[row['ID2']] +=1
                dict_of_winning_comparaison[row['ID2']] +=[row['ID1']]  
            else:
                id_to_target[row['ID2']] =2
                dict_of_winning_comparaison[row['ID2']] =[row['ID1']] 
        # else:  # Comparison == 0.5
            # id_to_target[row['ID1']] = row['ID1']
            # id_to_target[row['ID2']] = row['ID2']
            
    for id1 in dict_of_winning_comparaison.keys():
        for id2 in dict_of_winning_comparaison[id1]:
            if id_to_target[id2]>id_to_target[id1]:
                id_to_target[id1]=id_to_target[id2]

    # print(id_to_target)
    # print(unique_ids)

    # Create a list of original data tuples using the id_to_target mapping
    original_data = [(id, id_to_target[id]) for id in unique_ids]

    # Create the reconstructed original DataFrame
    original_df = pd.DataFrame(original_data, columns=['ID', 'TARGET'])

    return original_df






def get_training_pairwise(df):
    
    pairs = list(itertools.combinations(df.values, 2))
    # print(len(pairs))
    # print(pairs[0])
    pairwise_data = []

    
    for x,y in pairs:
        
        pairwise_data.append(np.hstack((x,y)))

    # print(len(pairwise_data)==len(pairs))
    # print(pairwise_data[0])
    
    pairwise_df = pd.DataFrame(pairwise_data, columns=np.hstack((df.columns+'1',df.columns+'2')))
    new_order=[column  for column_pair in [(f"{column}1",f"{column}2") for column in df.columns ] for column in column_pair ]
    
    return pairwise_df[new_order]
    

if __name__=='__main__':
    # data = {'ID': [1, 2, 3, 4, 5],
    #     'TARGET': [1, 4, 3, 3, 5]}
    # df = pd.DataFrame(data)

    # # Create pairwise dataset
    # pairwise_df = create_pairwise_dataset(df)

    # # Print pairs of Target values
    # print(pairwise_df)
    
    # print(reconstruct_original_dataset(pairwise_df))
    
    # pairwise_data = {'ID1': [1, 1, 2],
    #              'ID2': [2, 3, 3],
    #              'Comparison': [0.3, 0.8, 0.2]}
    # pairwise_df = pd.DataFrame(pairwise_data)

    # # Reconstruct the original dataset
    # print(reconstruct_original_dataset(pairwise_df))
    
    
    num_entries = 100
    fake_data = {
        'ID': [i for i in range(1, num_entries + 1)],
        'Date': pd.date_range(start='2023-01-01', periods=num_entries, freq='D'),
        'Weather': np.random.choice([0,1,2], size=num_entries),
        'Temperature': np.random.uniform(10, 30, size=num_entries)  # Random temperature between 10°C and 30°C
    }

    df = pd.DataFrame(fake_data)
    print(get_training_pairwise(df))
    

    
    
