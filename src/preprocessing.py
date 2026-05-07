import pandas as pd
from sklearn.model_selection import train_test_split


class InsurancePreprocessing:

    def __init__(self,df):
        self.df = df

    def clean_data(self):
        
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].str.strip()
        return self.df

    def encode_data(self):
        self.df['sex']=self.df['sex'].map({'female':0,'male':1})   
        self.df['smoker']= self.df['smoker'].map({'no':0,'yes':1})
        self.df = pd.get_dummies(self.df,columns=['region'],prefix='reg')
        return self.df

    def split_data(self,target_column='charges'):
        X = self.df.drop(target_column,axis=1)
        y = self.df[target_column]
        return train_test_split(X,y,test_size=0.2,random_state=42)
       