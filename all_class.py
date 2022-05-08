# from itertools import combinations
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn.base import BaseEstimator
# from sklearn.base import ClassifierMixin
# from sklearn.preprocessing import LabelEncoder
# from sklearn.base import clone
# from sklearn.pipeline import _name_estimators
# import numpy as np
# import operator
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import cross_val_score
# import numpy as np
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# class SBS():
#     def __init__(self, estimator, k_features,scoring=accuracy_score,test_size=0.25, random_state=1):
#         self.scoring = scoring
#         self.estimator = clone(estimator)
#         self.k_features = k_features
#         self.test_size = test_size
#         self.random_state = random_state
#     def fit(self, X, y):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,random_state=self.random_state)
#         dim = X_train.shape[1]
#         self.indices_ = tuple(range(dim))
#         self.subsets_ = [self.indices_]
#         score = self._calc_score(X_train, y_train,X_test, y_test, self.indices_)
#         self.scores_ = [score]
#         while dim > self.k_features:
#             scores = []
#             subsets = []
#             for p in combinations(self.indices_, r=dim - 1):
#                 score = self._calc_score(X_train, y_train,X_test, y_test, p)
#                 scores.append(score)
#                 subsets.append(p)
#             best = np.argmax(scores)
#             self.indices_ = subsets[best]
#             self.subsets_.append(self.indices_)
#             dim -= 1
#             self.scores_.append(scores[best])
#         self.k_score_ = self.scores_[-1]
#         return self
#     def transform(self, X):
#         return X[:, self.indices_]
#     def _calc_score(self, X_train, y_train, X_test, y_test, indices):
#         self.estimator.fit(X_train[:, indices], y_train)
#         y_pred = self.estimator.predict(X_test[:, indices])
#         score = self.scoring(y_test, y_pred)
#         return score

def plot_cultivars(Feature,Wavelength,Label,Index,year,Cultivar='3C'):
    if Cultivar=='All':
        Data_mature=np.mean(Feature[Label==1],axis=0)
        Data_immature=np.mean(Feature[Label==0],axis=0)
        plt.figure(figsize=(10,8))
        plt.title('Peanut Maturity Vs Immaturity')
        plt.plot(Wavelength,Data_mature,'-b')
        plt.plot(Wavelength,Data_immature,'-r')
        plt.xticks(Wavelength[::20], Wavelength[::20], rotation ='vertical')
        plt.legend(['Mature','Immature'],loc ="upper right")
#         plt.show()
        plt.savefig(f'MatureVsImmature_{year}.jpg',dpi=600)
    else:    
        idx=np.where(Index==Cultivar)[0][0]
        Data_3C=Feature[idx*15:(idx+1)*15]
        Label_3C=Label[idx*15:(idx+1)*15]
        Data_3C_M=np.mean(Data_3C[Label_3C==1],axis=0)
        Data_3C_IM=np.mean(Data_3C[Label_3C==0],axis=0)
        plt.figure(figsize=(10,8))
        plt.title(f'Spectrum of Cultivars {Cultivar}')
        plt.plot(Wavelength,Data_3C_M,'-b')
        plt.xticks(Wavelength[::20], Wavelength[::20], rotation ='vertical')
        plt.plot(Wavelength,Data_3C_IM,'-r')
        plt.xticks(Wavelength[::20], Wavelength[::20], rotation ='vertical')
#         plt.show()
        plt.savefig(f'MatureVsImmature_{Cultivar}_{year}.jpg',dpi=600)

def datacreate_2016(file='C:/Users/tusha/Downloads/Peanut_Maturity.csv',path1='C:/All/Peanut_Maturity_Classification',W_select=205):
    """"
    Input: CSV file of Peanut_Maturity having columns Group and Plot,
    Output: Index: Index of all peanuts specifies (1A, 2A,.....)
    Maturity: Matutre:1, Immature: 0
    Feature: DataFrame of 2016 and 2017 spectrum of all peanuts
    """
    DataFrame_2016=pd.read_csv(file).dropna(axis=0,how='all')
    Index_2016=DataFrame_2016['Group'][0:-1:15].values
    Maturity_2016=DataFrame_2016['Marure/Immature'].map({'M':1,'IM':0}).values

    Filename1=[os.path.join(path1,f'Spectral_{i}_1-15_Side1.csv') for i in Index_2016]
    Filename2=[os.path.join(path1,f'Spectral_{i}_1-15_Side2.csv') for i in Index_2016]
    Data=pd.read_csv(Filename1[0]).add(pd.read_csv(Filename2[0]))
    for i in range(1,len(Filename1)):
        Data=pd.concat([Data,pd.read_csv(Filename1[i]).add(pd.read_csv(Filename2[i]))],axis=0)
    Data=Data.drop(columns=['Unnamed: 0'])
    All_columns=Data.columns
    Final_Data_2016=Data.values/(np.sum(Data,axis=1).values.reshape(-1,1))
    Final_Data_Selected_2016=Final_Data_2016[:,W_select:]
    return Index_2016,All_columns,Maturity_2016,Final_Data_2016,Final_Data_Selected_2016

def datacreate_2017(file='C:/Users/tusha/Downloads/Peanut_Maturity_.csv',path2='C:/All/Peanut_Maturity_Classification_2017',W_select=205):
    """"
    Input: CSV file of Peanut_Maturity having columns Group and Plot,
    Output: Index: Index of all peanuts specifies (1A, 2A,.....)
    Maturity: Matutre:1, Immature: 0
   Feature: DataFrame of 2016 spectrum of all peanuts
    """
    DataFrame_2017=pd.read_csv(file).dropna(axis=0,how='all')
    Index_2017=DataFrame_2017['Plot '][0:-1:15].values
    Maturity_2017=DataFrame_2017['Mesocarp'].map({'Black':1, 'Brown':1, 'Orange':0, 'Yellow':0}).values
    Filename=[os.path.join(path2,f'Spectral_{i}1_20170921_.csv') for i in Index_2017]
    Data1=pd.read_csv(Filename[0])
    for i in range(1,len(Filename)):
        Data1=pd.concat([Data1,pd.read_csv(Filename[i])],axis=0)
    Data=Data1.drop(columns=['Unnamed: 0'])
    Final_Data_2017=Data.values/(np.sum(Data,axis=1).values.reshape(-1,1))
    Final_Data_Selected_2017=Final_Data_2017[:,W_select:]
    return Index_2017,Maturity_2017,Final_Data_2017,Final_Data_Selected_2017
def train_test_dataset_create(X_train,X_test,y_train,y_test,Features=35):
#Feature_Selection
    """
    Select Features and Creating training-testing data 
    """
    Data_diff=(np.mean(X_train[y_train==1],axis=0)-np.mean(X_train[y_train==0],axis=0))
    pos_Index,neg_Index=np.where(Data_diff>0)[0],np.where(Data_diff<0)[0]
    Pos_Data_diff,Neg_Data_diff=Data_diff[pos_Index],Data_diff[neg_Index]
    All_index=np.zeros(Features*2,dtype='int')
    All_index[0:Features],All_index[Features:]=pos_Index[np.argsort(Pos_Data_diff)[-Features:]],\
    neg_Index[np.argsort(Neg_Data_diff)[-Features:]]
    #Data splitting
    X_train, X_test = X_train[:,All_index],X_test[:,All_index]
    return X_train,X_test,y_train,y_test   
def specificity(Confusion_Matrix):
    tn, fp, fn, tp = Confusion_Matrix.ravel()
    return tn/(tn+fp)

# def wavelength_channel_conversion(X,option=1):
#         lemda1=400.450400;
#         lemda2=401.673000;
#         if option==1:
#             return np.int16((X-lemda1)/(lemda2-lemda1))+1;
#         elif option==0:
#             return (lemda2-lemda1)*X+lemda1;
#         else:
#             raise TypeError("Only 0 or 1 is allowed")