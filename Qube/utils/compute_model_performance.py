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


def process(df,percentage):
    sub_df=select_precentage_of_df(df,percentage)
    print(len(sub_df))
    print(len(df))
    sub_df['Comparison']= np.random.choice([0, 2], size=len(sub_df), p=[.5,.5])
    df2 = pd.merge(df, sub_df, on=['ID1','ID2'], how='left', suffixes=('', '_modified'))
    # Replace the original 'Value' column with the modified values
    df2['Comparison'] = df2['Comparison_modified'].fillna(df2['Comparison']).astype(int)
    df2.drop(['Comparison_modified'], axis=1, inplace=True)
    
    print("accuract between df and df scrambled is ",accuracy(df2['Comparison'],df['Comparison']))
    return df2


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
    

def mean_error(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length")
    print("list length is", len(list1))
    squared_errors = [(x - y) for x, y in zip(list1, list2)]
    mse = sum(squared_errors) / len(list1)
    return mse 

def verify(x,y):
    if (x> (1.1) and y==2) or (x< (0.9) and y==0) or (x<(1.1) and x>(0.9) and y==1) :
        return 1
    else:
        return 0

def accuracy(list1,list2):
    metrics=[sum([verify(x,y) for x, y in zip(list1, list2)])/ len(list1),
            (sum([verify(x,y) for x, y in zip(list1, list2) if y==0])/ len([y for y in list2 if y==0]),len([y for y in list2 if y==0])),
            (sum([verify(x,y) for x, y in zip(list1, list2) if y==2])/ len([y for y in list2 if y==2]),len([y for y in list2 if y==2]))]
        
    return metrics

class Model:
    def __init__(self,regressor_FR,regressor_DE,depth=None,n_estimators=None,max_iter=None,Solver=None):
        self.regressor_class_FR= regressor_FR
        self.regressor_class_DE= regressor_DE
        self.depth=depth
        self.estimators=n_estimators
        self.solver=Solver
        self.iters=max_iter
    def refresh_regressors(self):
        if self.depth  is None and self.solver is None:
            self.regressor_FR= self.regressor_class_FR()
            self.regressor_DE= self.regressor_class_DE()
        elif self.solver is None:
            self.regressor_FR= self.regressor_class_FR(max_depth=self.depth,n_estimators=self.estimators)
            self.regressor_DE= self.regressor_class_DE(max_depth=self.depth,n_estimators=self.estimators)
        if self.solver is not None:
            self.regressor_FR= self.regressor_class_FR(max_iter=self.iters,solver=self.solver)
    
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
        print("evaluate",100 *spearmanr(Y_Pred[columns], Y_test["TARGET"]).correlation)
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
    
    def Compute(self,iterations=20,columns=['TARGET','Rank'],custom_group=2,group=False,rank_before=True,split_countries=True,split=True,verbose=True):
        valid_list=[]
        valid_list_FR=[]
        valid_list_DE=[]
        train_list=[] 
        for i in range(iterations):
            if group :
                columns=handle_group_columns(columns,'Rank_group')
                
            if split :
                if not split_countries:
                    Y_test,Y_train,X_train_DE,Y_train_DE,X_train_FR,Y_train_FR,X_test_DE,Y_test_DE,X_test_FR,Y_test_FR=self.split_into_tests_before_countries(rank_before,custom_group,test_size=0.25)
                else:
                    Y_test,Y_train,X_train_DE,Y_train_DE,X_train_FR,Y_train_FR,X_test_DE,Y_test_DE,X_test_FR,Y_test_FR=self.split_into_countries_before_tests(rank_before,custom_group,test_size=0.25)


                self.refresh_regressors()
            
                self.fit(X_train_FR,Y_train_FR,X_train_DE,Y_train_DE,columns)
                

                pred_FR = self.predict_FR(X_test_FR)
            
                pred_DE = self.predict_DE(X_test_DE)
                
                Y_pred=self.reunite(pred_FR,pred_DE,X_test_FR,X_test_DE,columns,Y_test)
                

                validation=self.evaluate(Y_pred,Y_test,columns)
                valid_list_FR+=[evaluation(pred_FR,Y_test_FR)]
                valid_list_DE+=[evaluation(pred_DE,Y_test_DE)]
                
                
                training=self.evaluate(self.reunite(self.predict_FR(X_train_FR),self.predict_DE(X_train_DE),X_train_FR,X_train_DE,columns,Y_train),Y_train,columns)
                train_list+=[training]
            else :  

                X_train, X_test, Y_train, Y_test=train_test_split(self.X,self.Y, test_size=0.5)
                print("X_train and Y_train same ID ?",X_train['ID'].tolist()==Y_train['ID'].tolist())
                print("X_test and Y_test same ID ?",X_test['ID'].tolist()==Y_test['ID'].tolist())
                Y_train_pairwise=create_pairwise_dataset(Y_train)
                X_train_pairwise=get_training_pairwise(X_train)
                Y_test_pairwise=create_pairwise_dataset(Y_test)
                X_test_pairwise=get_training_pairwise(X_test)
                if verbose:
                    print("pairwise creation done")
                self.refresh_regressors()

                self.regressor_FR.fit(X_train_pairwise.drop(['ID1','ID2'],axis=1),(Y_train_pairwise['Comparison']))

                Y_pred_pairwise_list=self.regressor_FR.predict(X_test_pairwise.drop(['ID1','ID2'],axis=1))

                Y_train_pred_pairwise_list=self.regressor_FR.predict(X_train_pairwise.drop(['ID1','ID2'],axis=1))

                
                print("error_val",mean_error(Y_pred_pairwise_list, Y_test_pairwise['Comparison']))

                print('accuracy_val',accuracy(Y_pred_pairwise_list, Y_test_pairwise['Comparison']) )

                print("error_train",mean_error(Y_train_pred_pairwise_list, Y_train_pairwise['Comparison']))

                print('accuracy_train',accuracy(Y_train_pred_pairwise_list, Y_train_pairwise['Comparison']) )

                Y_pred_pairwise= pd.DataFrame({'ID1': X_test_pairwise["ID1"],'ID2': X_test_pairwise["ID2"],'Comparison': Y_pred_pairwise_list})

                Y_pred_train_pairwise= pd.DataFrame({'ID1': X_train_pairwise["ID1"],'ID2': X_train_pairwise["ID2"],'Comparison': Y_train_pred_pairwise_list})
                print(pd.concat((Y_pred_pairwise.reset_index(drop=True),Y_test_pairwise.reset_index(drop=True)),axis=1).head(1000))
                if verbose:
                    print("reconstructing rank ...")
                Y_pred=reconstruct_original_dataset(Y_pred_pairwise)
                Y_pred_train=reconstruct_original_dataset(Y_pred_train_pairwise)
                Y_pred=Y_pred.sort_values(by=['ID'], key=lambda x: x.map({k: i for i, k in enumerate(Y_test['ID'].tolist())}))
                Y_pred_train=Y_pred_train.sort_values(by=['ID'], key=lambda x: x.map({k: i for i, k in enumerate(Y_train['ID'].tolist())}))
                print(pd.concat((Y_pred.reset_index(drop=True),Y_test.reset_index(drop=True)),axis=1))
                print("Y_pred and Y_test same ID ?",Y_pred['ID'].tolist()==Y_test['ID'].tolist())
                print("Same length ?",len(Y_pred['ID'].tolist())==len(Y_test['ID'].tolist()))
                # print(pd.concat((Y_pred.reset_index(drop=True),Y_test.reset_index(drop=True)),axis=1))
                validation=self.evaluate(Y_pred,Y_test,['Rank_recreated'])[0]
                training=self.evaluate(Y_pred_train,Y_train,['Rank_recreated'])[0]
                print("validation" ,validation , "training" ,training)
                valid_list_DE=[np.nan]
                valid_list_FR=[np.nan]
                
            # print("training done")
            valid_list+=[validation]
            train_list+=[training]
            

        
        if isinstance(columns,list):
            for i,column in enumerate(columns):
                print( f"average {column} evaluation score at:", np.round(sum([row[i] for row in valid_list])/len(valid_list),1) , "   training at:", np.round(sum([row[i] for row in train_list])/len(train_list),1) , "    FR model at:", np.round(sum([row[i] for row in valid_list_FR])/len(valid_list_FR),1), "    DE model at:", np.round(sum([row[i] for row in valid_list_DE])/len(valid_list_DE),1))
                
        else:
            print( f"average {columns} evaluation score at ", sum(valid_list)/len(valid_list) , "training at :" ,  sum(train_list)/len(train_list),"training at :" ,  sum(valid_list_FR)/len(valid_list_FR),"training at :" ,  sum(valid_list_DE)/len(valid_list_DE) )
        
        
        print("\n")

    



