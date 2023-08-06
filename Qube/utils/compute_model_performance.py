from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
from typing import Optional
from .Dataset_modification import *

def average_stats(df,labels,regressor,columns=['TARGET','Rank'],iterations=20,depth=None,group=False,linear=False,custom_group=None):
    valid_list=[]
    train_list=[] 

    
    for i in range(iterations):
        if group :
            columns=handle_group_columns(columns,'Rank_group')
        
        X_train, X_test, Y_train, y_test = train_test_split(df, labels, test_size=0.25)
        Y_train['Rank']= Y_train['TARGET'].rank()
        if custom_group is not None:
                Y_train['Rank_group_'+str(custom_group)]=pd.qcut(Y_train.Rank,custom_group).cat.codes
                columns=handle_group_columns(columns,'Rank_group_'+str(custom_group))    
        
                
        if depth is None:
            DF = regressor()
        else:            
            DF = regressor(max_depth=depth)
        #   Fit the ridge regressor
        
        DF.fit(X_train, Y_train[columns])
        output_train = DF.predict(X_test)
        # print(output_train)
        validation=evaluation(output_train,y_test)
        training=evaluation(DF.predict(X_train),Y_train)
        valid_list+=[validation]
        train_list+=[training]
    print(columns)
    if isinstance(columns,list):
        for i,column in enumerate(columns):
            print( f"average {column} evaluation score at ", sum([row[i] for row in valid_list])/len(valid_list) , "training at ", sum([row[i] for row in train_list])/len(train_list) )
    else:
        print( f"average {columns} evaluation score at ", sum(valid_list)/len(valid_list) , "training at :" ,  sum(train_list)/len(train_list) )
    
    print("\n")





def handle_group_columns(columns,Rank_group):
    if columns is None:
        columns= Rank_group
    elif Rank_group  not in columns: 
        if isinstance(columns,list):
            columns=columns+[Rank_group]
        else:
            columns=[columns]+[Rank_group]
    print("columns are ", columns)
    return columns

def evaluation(output,df):
    term=output[0]
    if not isinstance(term, np.ndarray):
        return  100 *spearmanr(output, df["TARGET"]).correlation
    else:
        return  [100 *spearmanr(output[:,i], df["TARGET"]).correlation for i in range(len(term)) ]
    



class Model:
    def __init__(self,regressor_FR,regressor_DE):
        self.regressor_class_FR= regressor_FR
        self.regressor_class_DE= regressor_DE

    def refresh_regressors(self):
        self.regressor_FR= self.regressor_class_FR()
        self.regressor_DE= self.regressor_class_DE()
    
    def get_data_and_order(self,X,Y,custom_group=2):
        self.X=X
        self.Y=Y
        self.Y['Rank']= self.Y['TARGET'].rank().astype(int)
        self.Y['Rank_group']=pd.qcut(self.Y.Rank,custom_group).cat.codes
        

    def split_datasets(self,X, Y):
        return seperate_data_by_countries(X,Y)
    
    def predict_FR(self,X):
        return self.regressor_FR.predict(X)
    def predict_DE(self,X):
        return self.regressor_DE.predict(X)
    
    def reunite(self,pred_FR,pred_DE):
        return pd.concat([pred_FR,pred_DE]).sort_index()
    
    def evaluate(self,Pred,Y_test):
        return evaluation(Pred,Y_test)
    
    def Compute(self,iterations=20,columns=['TARGET','Rank'],group=False):
        valid_list=[]
        train_list=[] 
        for i in range(iterations):
            if group :
                columns=handle_group_columns(columns,'Rank_group')
                
             

            
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.25)

            
            X_train_DE,Y_train_DE,X_train_FR,Y_train_FR=seperate_data_by_countries(X_train,Y_train,rank=False)
            X_test_DE,Y_test_DE,X_test_FR,Y_test_FR=seperate_data_by_countries(X_test,Y_test,rank=False)
            # print(X_train_FR.info())
            # print(Y_train_FR.info())
            # print(X_test_FR.info())
            # print(X_test_FR.info())
            self.refresh_regressors()
            self.regressor_FR.fit(X_train_FR, Y_train_FR[columns])
            self.regressor_DE.fit(X_train_DE, Y_train_DE[columns])
            
            pred_FR = self.regressor_FR.predict(X_test_FR)
            pred_DE = self.regressor_DE.predict(X_test_DE)
            
            
            Y_pred_FR = pd.DataFrame(index=X_test_FR.index)
            Y_pred_DE = pd.DataFrame(index=X_test_DE.index)
            print(Y_pred_FR)
            for i,column in enumerate(columns):
                Y_pred_FR[column]=pred_FR[:,i]
                Y_pred_DE[column]=pred_DE[:,i]

            Y_pred=self.reunite(Y_pred_FR,Y_pred_DE)
            validation=evaluation(Y_pred,Y_test)
            training=evaluation(self.reunite(self.predict_FR(X_train_FR),self.predict_DE(X_train_DE)),Y_train)

            valid_list+=[validation]
            train_list+=[training]

        
        if isinstance(columns,list):
            for i,column in enumerate(columns):
                print( f"average {column} evaluation score at ", sum([row[i] for row in valid_list])/len(valid_list) , "training at ", sum([row[i] for row in train_list])/len(train_list) )
        else:
            print( f"average {columns} evaluation score at ", sum(valid_list)/len(valid_list) , "training at :" ,  sum(train_list)/len(train_list) )
        
        print("\n")

    



