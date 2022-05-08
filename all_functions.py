import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def plot_cultivars(Feature,Wavelength,Label,Index,year,Cultivar='3C'):
    if Cultivar=='All':
        Data_mature=np.mean(Feature[Label==1],axis=0)
        Data_immature=np.mean(Feature[Label==0],axis=0)
        plt.figure(figsize=(10,8))
        plt.title('Peanut Maturity Vs Immaturity')
        plt.plot(Wavelength,Data_mature,'-b')
        plt.plot(Wavelength,Data_immature,'-r')
        plt.xticks(Wavelength[::20], Wavelength[::20], rotation ='vertical')
        plt.legend(['Mature','Immature'],loc ="upper left")
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
        #plt.xticks(Wavelength[::20], Wavelength[::20], rotation ='vertical')
        plt.plot(Wavelength,Data_3C_IM,'-r')
        plt.xticks(Wavelength[::20], Wavelength[::20], rotation ='vertical')
        plt.legend(['Mature','Immature'],loc ="upper left")
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
    Color_peanut=DataFrame_2016['Color (M:black, brown/ IM: orange, yellow)'];
    Color_peanut=Color_peanut.map({'black':0,'brown':1,'orange':2, 'smashed /yellow':3, 'yellow':3}).values
    Color_index=[np.where(Color_peanut==i)[0] for i in range(4)]

    Filename1=[os.path.join(path1,f'Spectral_{i}_1-15_Side1.csv') for i in Index_2016]
    Filename2=[os.path.join(path1,f'Spectral_{i}_1-15_Side2.csv') for i in Index_2016]
    Data=pd.read_csv(Filename1[0]).add(pd.read_csv(Filename2[0]))
    for i in range(1,len(Filename1)):
        Data=pd.concat([Data,pd.read_csv(Filename1[i]).add(pd.read_csv(Filename2[i]))],axis=0)
    Data=Data.drop(columns=['Unnamed: 0'])
    All_columns=Data.columns
    Final_Data_2016=Data.values/(np.sum(Data,axis=1).values.reshape(-1,1))
    Final_Data_Selected_2016=Final_Data_2016[:,W_select:]
    return Index_2016,Color_index,All_columns,Maturity_2016,Final_Data_2016,Final_Data_Selected_2016

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
  
    Color_peanut=DataFrame_2017['Mesocarp'].map({'Black':0, 'Brown':1, 'Orange':2, 'Yellow':3}).values
    Color_index=[np.where(Color_peanut==i)[0] for i in range(4)]
    
    Filename=[os.path.join(path2,f'Spectral_{i}1_20170921_.csv') for i in Index_2017]
    Data1=pd.read_csv(Filename[0])
    for i in range(1,len(Filename)):
        Data1=pd.concat([Data1,pd.read_csv(Filename[i])],axis=0)
    Data=Data1.drop(columns=['Unnamed: 0'])
    Final_Data_2017=Data.values/(np.sum(Data,axis=1).values.reshape(-1,1))
    Final_Data_Selected_2017=Final_Data_2017[:,W_select:]
    return Index_2017,Color_index,Maturity_2017,Final_Data_2017,Final_Data_Selected_2017
def Strong_feature_train_test_dataset_create(X_train,X_test,y_train,y_test,Features=35):
#Feature_Selection
    """
    Select Features and Creating training-testing data 
    """
    Data_diff=(np.mean(X_train[y_train==1],axis=0)-np.mean(X_train[y_train==0],axis=0))
    #Data_diff2=(np.mean(X_test[y_test==1],axis=0)-np.mean(X_test[y_test==0],axis=0))
    #Data_diff=(Data_diff1+Data_diff2)
    pos_Index,neg_Index=np.where(Data_diff>0)[0],np.where(Data_diff<0)[0]
    Pos_Data_diff,Neg_Data_diff=Data_diff[pos_Index],Data_diff[neg_Index]
    All_index=np.zeros(Features*2,dtype='int')
    All_index[0:Features],All_index[Features:]=pos_Index[np.argsort(Pos_Data_diff)[-Features:]],\
    neg_Index[np.argsort(Neg_Data_diff)[-Features:]]
    #Data splitting
    X_train, X_test = X_train[:,All_index],X_test[:,All_index]
    return X_train,X_test,y_train,y_test 


def anova_train_test_dataset_create(X_train,X_test,y_train,y_test,Features=35):
    """https://machinelearningmastery.com/feature-selection-with-numerical-input-data/"""
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k=2*Features)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs,X_test_fs,y_train,y_test 

def specificity(Confusion_Matrix):
    tn, fp, fn, tp = Confusion_Matrix.ravel()
    return tn/(tn+fp)