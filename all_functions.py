import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

W=np.array([400.450400,401.673000, 402.896000, 404.119400, 405.343300, 406.567600, 407.792400, 409.017500, 
410.243100, 411.469100, 412.695500, 413.922300, 415.149600, 416.377300, 417.605300, 
418.833900, 420.062800, 421.292200, 422.521900, 423.752100, 424.982700, 426.213700, 
427.445100, 428.676900, 429.909100, 431.141700, 432.374800, 433.608200, 434.842000, 
436.076300, 437.310900, 438.546000, 439.781400, 441.017300, 442.253500, 443.490100, 
444.727200, 445.964600, 447.202500, 448.440600, 449.679300, 450.918300, 452.157700, 
453.397400, 454.637600, 455.878200, 457.119100, 458.360400, 459.602100, 460.844200, 
462.086700, 463.329600, 464.572800, 465.816400, 467.060400, 468.304700, 469.549500, 
470.794600, 472.040200, 473.286000, 474.532200, 475.778900, 477.025800, 478.273200, 
479.520900, 480.769000, 482.017500, 483.266300, 484.515400, 485.765000, 487.014900, 
488.265200, 489.515800, 490.766800, 492.018100, 493.269800, 494.521900, 495.774300, 
497.027100, 498.280200, 499.533700, 500.787500, 502.041700, 503.296200, 504.551100, 
505.806300, 507.061800, 508.317700, 509.574000, 510.830600, 512.087500, 513.344800, 
514.602400, 515.860400, 517.118700, 518.377300, 519.636400, 520.895600, 522.155300, 
523.415300, 524.675500, 525.936200, 527.197100, 528.458500, 529.720100, 530.982100, 
532.244300, 533.507000, 534.769900, 536.033100, 537.296700, 538.560500, 539.824800, 
541.089400, 542.354200, 543.619400, 544.884900, 546.150600, 547.416800, 548.683200, 
549.950000, 551.217000, 552.484400, 553.752100, 555.020100, 556.288300, 557.556900, 
558.825900, 560.095100, 561.364600, 562.634400, 563.904500, 565.174900, 566.445700, 
567.716700, 568.988000, 570.259600, 571.531500, 572.803700, 574.076200, 575.349000, 
576.622100, 577.895400, 579.169100, 580.443000, 581.717300, 582.991800, 584.266600, 
585.541700, 586.817000, 588.092800, 589.368700, 590.644900, 591.921400, 593.198100, 
594.475200, 595.752600, 597.030200, 598.308000, 599.586200, 600.864600, 602.143300, 
603.422300, 604.701500, 605.981100, 607.260900, 608.540900, 609.821200, 611.101700, 
612.382600, 613.663700, 614.945100, 616.226700, 617.508500, 618.790700, 620.073100, 
621.355800, 622.638700, 623.921900, 625.205300, 626.488900, 627.772800, 629.057000, 
630.341400, 631.626100, 632.911100, 634.196300, 635.481700, 636.767300, 638.053200, 
639.339400, 640.625700, 641.912400, 643.199200, 644.486300, 645.773700, 647.061300, 
648.349100, 649.637200, 650.925400, 652.214000, 653.502700, 654.791700, 656.080900, 
657.370400, 658.660000, 659.950000, 661.240100, 662.530400, 663.821000, 665.111800, 
666.402800, 667.694100, 668.985500, 670.277200, 671.569100, 672.861200, 674.153600, 
675.446000, 676.738900, 678.031900, 679.325000, 680.618400, 681.912000, 683.205800, 
684.499800, 685.794100, 687.088500, 688.383200, 689.678000, 690.973000, 692.268300, 
693.563700, 694.859400, 696.155200, 697.451300, 698.747600, 700.043900, 701.340600, 
702.637500, 703.934400, 705.231700, 706.529100, 707.826700, 709.124500, 710.422400, 
711.720600, 713.018900, 714.317500, 715.616200, 716.915100, 718.214200, 719.513400, 
720.812900, 722.112500, 723.412400, 724.712300, 726.012500, 727.312700, 728.613300, 
729.914000, 731.214800, 732.515900, 733.817100, 735.118400, 736.419900, 737.721700, 
739.023500, 740.325600, 741.627700, 742.930100, 744.232700, 745.535300, 746.838100, 
748.141100, 749.444300, 750.747600, 752.051100, 753.354700, 754.658500, 755.962400, 
757.266500, 758.570800, 759.875200, 761.179700, 762.484400, 763.789200, 765.094200, 
766.399400, 767.704600, 769.010000, 770.315600, 771.621300, 772.927100, 774.233200, 
775.539200, 776.845500, 778.151900, 779.458400, 780.765100, 782.071900, 783.378800, 
784.685900, 785.993000, 787.300400, 788.607800, 789.915400, 791.223100, 792.530900, 
793.838900, 795.147000, 796.455200, 797.763500, 799.071900, 800.380500, 801.689200, 
802.998000, 804.306900, 805.615900, 806.925000, 808.234300, 809.543700, 810.853100, 
812.162700, 813.472500, 814.782300, 816.092200, 817.402200, 818.712400, 820.022600, 
821.332900, 822.643400, 823.953900, 825.264600, 826.575300, 827.886200, 829.197100, 
830.508200, 831.819300, 833.130500, 834.441900, 835.753300, 837.064800, 838.376300, 
839.688000, 840.999900, 842.311700, 843.623700, 844.935700, 846.247800, 847.560100, 
848.872300, 850.184700, 851.497100, 852.809700, 854.122300, 855.435000, 856.747700, 
858.060500, 859.373500, 860.686500, 861.999500, 863.312700, 864.625900, 865.939200, 
867.252600, 868.565900, 869.879400, 871.193000, 872.506600, 873.820300, 875.134000, 
876.447800, 877.761700, 879.075600, 880.389500, 881.703600, 883.017700, 884.331800, 
885.646100, 886.960300, 888.274700, 889.589000, 890.903400, 892.217900, 893.532400, 
894.846900, 896.161600, 897.476300, 898.791000, 900.105700, 901.420500, 902.735400, 
904.050200, 905.365100, 906.680100, 907.995000, 909.310100, 910.625100, 911.940200, 
913.255400, 914.570600, 915.885700, 917.200900, 918.516200, 919.831500, 921.146900, 
922.462200, 923.777600, 925.093000, 926.408400, 927.723800, 929.039300, 930.354700, 
931.670300, 932.985800, 934.301400, 935.616900, 936.932500, 938.248000, 939.563700, 
940.879300, 942.194900, 943.510600, 944.826200, 946.141900, 947.457600, 948.773300, 
950.088900, 951.404700, 952.720300, 954.036000, 955.351700, 956.667400, 957.983200, 
959.298800, 960.614600, 961.930200, 963.246000, 964.561600, 965.877300, 967.193000, 
968.508700, 969.824300, 971.140000, 972.455600, 973.771200, 975.086900, 976.402500, 
977.718100, 979.033700, 980.349200, 981.664800, 982.980200, 984.295800, 985.611300, 
986.926700, 988.242200, 989.557600, 990.872900, 992.188400, 993.503700, 994.819000, 
996.134300, 997.449500, 998.764800, 1000.080000]).astype('int')

def plot_cultivars(Feature1,Feature2,Label1,Label2,Index,year,Cultivar='3C',Wavelength=W):
    if Cultivar=='All':
        Data_mature1=np.mean(Feature1[Label1==1],axis=0)
        Data_immature1=np.mean(Feature1[Label1==0],axis=0)
        Data_mature1=np.mean(Feature1[Label1==1],axis=0)
        Data_immature1=np.mean(Feature1[Label1==0],axis=0)
        
        Data_mature2=np.mean(Feature2[Label2==1],axis=0)
        Data_immature2=np.mean(Feature2[Label2==0],axis=0) 
        Data_mature2=np.mean(Feature2[Label2==1],axis=0)
        Data_immature2=np.mean(Feature2[Label2==0],axis=0)
        
        plt.figure(figsize=(10,8))
        #plt.title('Peanut Maturity Vs Immaturity',fontsize=24)
        plt.plot(Wavelength,Data_mature1,'-b')
        plt.plot(Wavelength,Data_immature1,'-r')
        plt.plot(Wavelength,Data_mature2,'-b')
        plt.plot(Wavelength,Data_immature2,'-r')
        
        plt.xticks(Wavelength[::60], Wavelength[::60], rotation ='vertical',fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Wavelength',fontsize=18)
        plt.ylabel('Relectance',fontsize=18)           
        plt.legend(['Mature','Immature'],loc ="upper left",fontsize=18)
#         plt.show()
        plt.savefig(f'MatureVsImmature_{year}.jpg',dpi=600)
    else:    
        idx=np.where(Index==Cultivar)[0][0]
        Data_3C1=Feature1[idx*15:(idx+1)*15]
        Label_3C1=Label1[idx*15:(idx+1)*15]
        Data_3C_M1=np.mean(Data_3C1[Label_3C1==1],axis=0)
        Data_3C_IM1=np.mean(Data_3C1[Label_3C1==0],axis=0)
        
        idx=np.where(Index==Cultivar)[0][0]
        Data_3C2=Feature2[idx*15:(idx+1)*15]
        Label_3C2=Label2[idx*15:(idx+1)*15]
        Data_3C_M2=np.mean(Data_3C2[Label_3C2==1],axis=0)
        Data_3C_IM2=np.mean(Data_3C2[Label_3C2==0],axis=0)
        
        Data_3C_M=(Data_3C_M1+Data_3C_M2)/2
        Data_3C_IM=(Data_3C_IM1+Data_3C_IM2)/2
        
        plt.figure(figsize=(10,8))
        #plt.title(f'Spectrum of Cultivars {Cultivar}')
        plt.plot(Wavelength,Data_3C_M,'-b')
        plt.plot(Wavelength,Data_3C_IM,'-r')
        
        
        plt.xticks(Wavelength[::60], Wavelength[::60], rotation ='vertical',fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Wavelength',fontsize=18)
        plt.ylabel('Relectance',fontsize=18)           
        plt.legend(['Mature','Immature'],loc ="upper left",fontsize=18)
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

def Max_feature_train_test_dataset_create(X_train,X_test,y_train,y_test,Features=35):
#Feature_Selection
    """
    Select Features and Creating training-testing data 
    """
    Data_diff=(np.mean(X_train[y_train==1],axis=0)-np.mean(X_train[y_train==0],axis=0))
    All_index=np.argsort(abs(Data_diff))[-2*Features:]
    X_train, X_test = X_train[:,All_index],X_test[:,All_index]
    return X_train,X_test,y_train,y_test 

def specificity(Confusion_Matrix):
    tp, fn, fp, tn = Confusion_Matrix.ravel()
    return tn/(tn+fp)