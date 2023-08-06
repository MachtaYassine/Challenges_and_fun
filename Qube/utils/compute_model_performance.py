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
        print("columns are ", columns)
    elif Rank_group  not in columns: 
        if isinstance(columns,list):
            columns=columns+[Rank_group]
            print("columns are ", columns)
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
    def __init__(self,regressor_FR,regressor_DE,depth=None):
        self.regressor_class_FR= regressor_FR
        self.regressor_class_DE= regressor_DE
        self.depth=depth

    def refresh_regressors(self):
        if self.depth is None:
            self.regressor_FR= self.regressor_class_FR()
            self.regressor_DE= self.regressor_class_DE()
        else :
            self.regressor_FR= self.regressor_class_FR(self.depth)
            self.regressor_DE= self.regressor_class_DE(self.depth)
    
    def get_data_and_order(self,X,Y):
        self.X=X
        self.Y=Y
        
        

    def split_datasets(self,X, Y):
        return seperate_data_by_countries(X,Y)
    

    def fit (self,X_FR,Y_FR,X_DE,Y_DE,columns):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.regressor_FR.fit(X_FR.drop('ID',axis=1), Y_FR[columns])
            self.regressor_DE.fit(X_DE.drop('ID',axis=1), Y_DE[columns])

    def predict_FR(self,X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.regressor_FR.predict(X.drop('ID',axis=1))
    def predict_DE(self,X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.regressor_DE.predict(X.drop('ID',axis=1))
    
    def reunite(self,pred_FR,pred_DE,X_test_FR,X_test_DE,columns,Y_test):
        Y_pred_FR = pd.DataFrame({'ID': X_test_FR["ID"]})
        Y_pred_DE = pd.DataFrame({'ID': X_test_DE["ID"]})
        for i,column in enumerate(columns):
                Y_pred_FR[column]=pred_FR[:,i]
                Y_pred_DE[column]=pred_DE[:,i]
        # print("Y_pred_FR \n",Y_pred_FR)
        # print("Y_pred_DE \n",Y_pred_DE)
        return (pd.concat([Y_pred_FR,Y_pred_DE]).sort_values(by=['ID'], key=lambda x: x.map({k: i for i, k in enumerate(Y_test['ID'].tolist())})))

    
    def evaluate(self,Y_Pred,Y_test,columns):
        return  [100 *spearmanr(Y_Pred[column], Y_test["TARGET"]).correlation for column in columns ]
        

    def split_into_tests_before_countries(self,rank_before,custom_group,test_size=0.25):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X.copy(deep=True), self.Y.copy(deep=True), test_size=test_size)
        X_train_DE,Y_train_DE,X_train_FR,Y_train_FR=seperate_data_by_countries(X_train,Y_train,rank_before_split=rank_before,custom_group=custom_group,keep_id=True)
        X_test_DE,Y_test_DE,X_test_FR,Y_test_FR=seperate_data_by_countries(X_test,Y_test,rank_before_split=True,custom_group=custom_group,keep_id=True)
        return Y_test,Y_train,X_train_DE,Y_train_DE,X_train_FR,Y_train_FR,X_test_DE,Y_test_DE,X_test_FR,Y_test_FR
    
    def split_into_countries_before_tests(self,rank_before,custom_group,test_size=0.25):
        X_DE,Y_DE,X_FR,Y_FR=seperate_data_by_countries(self.X.copy(deep=True),self.Y.copy(deep=True),rank_before_split=rank_before,custom_group=custom_group,keep_id=True)
        X_train_DE, X_test_DE, Y_train_DE, Y_test_DE = train_test_split(X_DE,Y_DE, test_size=test_size)
        X_train_FR, X_test_FR, Y_train_FR, Y_test_FR = train_test_split(X_FR, Y_FR, test_size=test_size)

        
        
        Y_test= pd.concat([Y_test_FR, Y_test_DE], axis=0)
        Test_Ids= [x for x in self.Y['ID'].tolist() if x in Y_test['ID'].tolist()]
        Y_test=Y_test.sort_values(by=['ID'], key=lambda x: x.map({k: i for i, k in enumerate(Test_Ids)}))

        Y_train= pd.concat([Y_train_DE, Y_train_FR], axis=0)
        
        Train_Ids= [x for x in self.Y['ID'].tolist() if x in Y_train['ID'].tolist()]
        Y_train=Y_train.sort_values(by=['ID'], key=lambda x: x.map({k: i for i, k in enumerate(Train_Ids)}))
        return Y_test,Y_train,X_train_DE,Y_train_DE,X_train_FR,Y_train_FR,X_test_DE,Y_test_DE,X_test_FR,Y_test_FR
    
    def Compute(self,iterations=20,columns=['TARGET','Rank'],custom_group=2,group=False,rank_before=True,split_countries=True):
        valid_list=[]
        valid_list_FR=[]
        valid_list_DE=[]
        train_list=[] 
        for i in range(iterations):
            if group :
                columns=handle_group_columns(columns,'Rank_group')
                
             
            if not split_countries:
                Y_test,Y_train,X_train_DE,Y_train_DE,X_train_FR,Y_train_FR,X_test_DE,Y_test_DE,X_test_FR,Y_test_FR=self.split_into_tests_before_countries(rank_before,custom_group,test_size=0.25)
            else:
                Y_test,Y_train,X_train_DE,Y_train_DE,X_train_FR,Y_train_FR,X_test_DE,Y_test_DE,X_test_FR,Y_test_FR=self.split_into_countries_before_tests(rank_before,custom_group,test_size=0.25)
                
            # print(X_train_FR.info())
            # print(Y_train_FR.info())
            # print(X_test_FR.info())
            # print(X_test_FR.info())
            self.refresh_regressors()
            
            self.fit(X_train_FR,Y_train_FR,X_train_DE,Y_train_DE,columns)
            

            pred_FR = self.predict_FR(X_test_FR)
            # print(pred_FR)
            pred_DE = self.predict_DE(X_test_DE)
            
            Y_pred=self.reunite(pred_FR,pred_DE,X_test_FR,X_test_DE,columns,Y_test)
            # print("Y_test \n",Y_test)
            # print("Y_pred \n",Y_pred)

            validation=self.evaluate(Y_pred,Y_test,columns)
            valid_list_FR+=[evaluation(pred_FR,Y_test_FR)]
            valid_list_DE+=[evaluation(pred_DE,Y_test_DE)]
            # print("eval done")
            training=self.evaluate(self.reunite(self.predict_FR(X_train_FR),self.predict_DE(X_train_DE),X_train_FR,X_train_DE,columns,Y_train),Y_train,columns)
            # print("training done")
            valid_list+=[validation]
            train_list+=[training]

        
        if isinstance(columns,list):
            for i,column in enumerate(columns):
                print( f"average {column} evaluation score at:", np.round(sum([row[i] for row in valid_list])/len(valid_list),1) , "   training at:", np.round(sum([row[i] for row in train_list])/len(train_list),1) , "    FR model at:", np.round(sum([row[i] for row in valid_list_FR])/len(valid_list_FR),1), "    DE model at:", np.round(sum([row[i] for row in valid_list_DE])/len(valid_list_DE),1))
                
        else:
            print( f"average {columns} evaluation score at ", sum(valid_list)/len(valid_list) , "training at :" ,  sum(train_list)/len(train_list) )
            print( f"average {column} evaluation score for FR_Dataset at ", sum([row[i] for row in valid_list_FR])/len(valid_list_FR), f"average {column} evaluation score for DE_Dataset at ", sum([row[i] for row in valid_list_DE])/len(valid_list_DE))
        
        print("\n")

    



