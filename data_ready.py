##1_import all library
import spectral.io.envi as envi #To read spectral data
import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

class hyp_image_processing:
    def __init__(self, hdr_file_name, hyp_file_name,file):
        self.hdr_file_name = hdr_file_name
        self.hyp_file_name = hyp_file_name
        self.file=file
    def open_envi_file(self,path):
        """
        Reading Envi file requires two files hdr and hyp

        hdr: Height,Width,Bands,Interleave,Quantization=32 bit, DataFormat=float32
        hyp: Reflectance of all bands
        envi_open creates an BilFile to read data, self.hyp_image
        """
        self.hyp_image = envi.open(os.path.join(path,self.hdr_file_name), os.path.join(path,self.hyp_file_name))    
    def show_specific_band(self,Number):
        """
        Number: Number of band that needs to read
        Show_image_for_specific_band and write the image
        """
        S_image=np.squeeze(self.hyp_image[:,:,Number]);
        filename1="Single_Wavelength for image "+str(self.file)+" Wavelength "+str(Number)+".jpg";
#         if plot_img1=="show":
        plt.figure(figsize=(10,8))
        plt.imshow(S_image,cmap='gray')
        plt.title(filename1[:-4]) 
#         if status=="Write":
        self.S_image_gray=((S_image/np.max(S_image))*255).astype(np.uint8)
        cv2.imwrite(filename1,self.S_image_gray) 
    def BGR_imread(self):
        """
        Create RGB image from hyp_image, self.hyp_image_BGR and show in matplotlib and write in opencv 
        """
        wave_length=np.array([450,550,650])
        BGR=wavelength_channel_conversion(wave_length,option=1)
        self.hyp_image_BGR=np.array(self.hyp_image[:,:,list(BGR)] ,dtype=np.uint8)
        filename2="RGB_image_"+str(self.file)+".jpg";
#         if plot_img=="show":
        plt.figure(figsize=(10,8))
        plt.imshow(cv2.cvtColor(self.hyp_image_BGR,cv2.COLOR_BGR2RGB))
        plt.title(filename2[:-4]) 
#         if status=="Write":
        RGBimage = cv2.cvtColor(self.hyp_image_BGR, cv2.COLOR_BGR2RGB)
        PILimage = Image.fromarray(RGBimage)
        PILimage.save(filename2, dpi=(600,600))
        #cv2.imwrite(filename2,self.hyp_image_BGR) 
    def image_cropped(self,crop=0.40):
        """
        Crop BGR image, self.hyp_image_BGR_cropped, show and write 
        """
        W=self.hyp_image_BGR.shape[1]
        self.hyp_image_BGR_cropped=self.hyp_image_BGR[:,int(0.40*W):,:]
        filename3="RGB_image_cropped for image "+str(self.file)+".jpg";
#         if plot_img=="show":
        plt.figure(figsize=(10,8))
        plt.imshow(cv2.cvtColor(self.hyp_image_BGR_cropped,cv2.COLOR_BGR2RGB))
        plt.title(filename3[:-4]) 
#         if status=="Write":
        cv2.imwrite(filename3,self.hyp_image_BGR_cropped)
    def ostu_thresholding(self):
        """
        Thresholding image using ostu thresholding, self.Image_ostu, show and write
        """
        Image_gray = cv2.cvtColor(self.hyp_image_BGR,cv2.COLOR_BGR2GRAY) #For ostu thresholding image has been converted to grayscale first    
        #Image_gray = cv2.GaussianBlur(Image_gray,(5,5),0)
        _, im_bw = cv2.threshold(Image_gray, 0, 255, cv2.THRESH_OTSU) #Binary Ostu threshold image
        self.Image_ostu=255-np.array(im_bw[:,int(im_bw.shape[1]*0.4):]) #Croped calibration panel
        filename4="Ostu_thresholded_image_"+str(self.file)+".jpg";
#         if plot_img=="show":
        plt.figure(figsize=(10,8))
        plt.imshow(self.Image_ostu,cmap='gray')
        plt.title(filename4[:-4]) 
#         if status=="Write":
        RGBimage = cv2.cvtColor(self.Image_ostu, cv2.COLOR_BGR2RGB)
        PILimage = Image.fromarray(RGBimage)
        PILimage.save(filename4, dpi=(600,600))
        #cv2.imwrite(filename4,self.Image_ostu)
    def Morphological_Erode(self,kernel = np.ones((12, 4), np.uint8)):
        """
        Removing Spot using Morphological Operation, show and write
        """
        Filtered_image_ostu = cv2.dilate(self.Image_ostu, kernel,iterations=1)
        self.Filtered_image_ostu = cv2.erode(Filtered_image_ostu, kernel,iterations=1)
        filename5="Morphologically_processed_image_"+str(self.file)+".jpg";
#         if plot_img=="show": 
        plt.figure(figsize=(10,8))
        plt.imshow(self.Filtered_image_ostu,cmap='gray')
        plt.title(filename5[:-4]) 
#         if status=="Write":
        RGBimage = cv2.cvtColor(self.Filtered_image_ostu, cv2.COLOR_BGR2RGB)
        PILimage = Image.fromarray(RGBimage)
        PILimage.save(filename5, dpi=(600,600))
        #cv2.imwrite(filename5, self.Filtered_image_ostu)
    def find_peanut_pixel_postion(self): 
        """
        Find all the pixel position where peanut,self.Pixel_Position, shape=(Peanut_Pixel_Number,2)
        Create object kmeans for finding 15 peanuts,self.kmeans
        Finding pixel position for all 15 peanuts individually, self.Pixel_Position, shape=(15,Peanut pixel number of ith peanut, 2)
        Find the index given by kmeans for all the peanuts, In (3,5) grid, it will start from (0,0). And self.Index will tell us
        what will be peanut label for (0,0), (0,1), ......
        """
        Pixel_Position=[(i,j) for j in range(self.Filtered_image_ostu.shape[1]) 
                        for i in range(self.Filtered_image_ostu.shape[0]) if self.Filtered_image_ostu[i][j]//255==1] 
        Peanut=15;
        self.Pixel_Position=np.array(Pixel_Position)
        self.kmeans = KMeans(n_clusters=Peanut, max_iter=500,tol=0.00001,random_state=0).fit(self.Pixel_Position);
        #Blank_image=identify_peanut_by_label(pixel_position,kmeans)
        self.Final_pixel=[self.Pixel_Position[np.where(self.kmeans.labels_==i)] for i in range(Peanut)];
        #Center of custering at by kmeans, C1
        custer_center=self.kmeans.cluster_centers_.astype(int)
        ##Cluster center from (0,0) to (2,4) in a (3,5),C1
        min_max_by_row=np.argsort(custer_center[:,0],axis=0);
        Max_Min_by_row=custer_center[min_max_by_row]
        A1=[];
        for i in range(3):
            A1.append(np.argsort(Max_Min_by_row[5*i:5*(i+1)][:,1])+5*i)
        Real_Max_Min=np.squeeze(Max_Min_by_row[np.array(A1).reshape(1,-1)]) 
        ##Generate C2 with respect to C1: Suppose C2[0]=C1[8], So, A=[8,....]
        A=[];
        for j in range(15):
            for i in range(15):
                if ((Real_Max_Min[:,0][j]==custer_center[:,0][i]) & (Real_Max_Min[:,1][j]==custer_center[:,1][i])):
                    A.append(i)
        self.Index=np.array(A) 
    def Peanut_Pixel_Reflectance(self):
        """
        Create an array Feature_Reflectance,self.Feature_Reflectance, Shape=(15,Peanut_pixel_number,Reflectance)
        """
        Feature_label=[];
        self.hyp_image_main=self.hyp_image[:,int(self.hyp_image.shape[1]*0.40):,:]
        for i in range(len(self.Final_pixel)):
            Feature=[];
            for k in range(len(self.Final_pixel[i])):
                Feature.append(self.hyp_image_main[self.Final_pixel[i][k][0],self.Final_pixel[i][k][1]])
            Feature_label.append(np.array(Feature)) 
        self.Feature_Reflectance=np.array(Feature_label,dtype=object)
    def identify_peanut_by_label(self):
        """
        Show all the peanuts with number identified, Should be equal to self.index
        """
        Colors=('red','green','blue')
        markers = ('s', 'x', 'o', '^', 'v')
        row=3;
        plt.figure(figsize=(4,5))
        filename6='Peanut_identification_'+str(self.file)+'.jpg'
        plt.scatter(self.Pixel_Position[:,1],self.Pixel_Position[:,0], c=self.kmeans.labels_, cmap='plasma')
        for i in range(5):
            plt.scatter(self.kmeans.cluster_centers_[i*row:(i+1)*row,1], self.kmeans.cluster_centers_[i*row:(i+1)*row,0], color=Colors[:row], marker=markers[i])
        plt.savefig(filename6,dpi=600)
    def Feature_Matrix_Creation(self):
        """
        Feature Matrix Creation, self.Final_Feature_Matrix, Shape=(15,467) 
        """
        self.Feature_Matrix=[np.mean(self.Feature_Reflectance[i],axis=0) for i in self.Index];
        #self.Final_Feature_Matrix=np.array(Feature_Matrix)[self.Index]
    def spatial_spectral_feature(self):
        hyp_parts_spec=[];
        for i in range(15):
            Peanut=self.Final_pixel[i];
            Peanut_r,Peanut_c=Peanut[:,0],Peanut[:,1];
            Pixel_min_max=np.min(Peanut_r),np.max(Peanut_r),np.min(Peanut_c),np.max(Peanut_c)
            Point_r=np.linspace(Pixel_min_max[0],Pixel_min_max[1],5).astype(int)
            Point_c=np.linspace(Pixel_min_max[2],Pixel_min_max[3],3).astype(int)
            hyp_spec=[];
            for k in range(2):
                for l in range(8):
                    Ind_Peanut=Peanut[np.squeeze(np.array([(Peanut_c>=Point_c[l]) & (Peanut_c<=Point_c[l+1]) & (Peanut_r>=Point_r[k]) & (Peanut_r<=Point_r[k+1])]))] 
                    hyp_spec.append(np.mean(self.hyp_image_main[Ind_Peanut[:,0],Ind_Peanut[:,1]],axis=0))
            hyp_parts_spec.append(hyp_spec)
        self.hyp_parts_spec=np.array(hyp_parts_spec);

def wavelength_channel_conversion(X,option=1):
        lemda1=400.450400;
        lemda2=401.673000;
        if option==1:
            return np.int16((X-lemda1)/(lemda2-lemda1))+1;
        elif option==0:
            return (lemda2-lemda1)*X+lemda1;
        else:
            raise TypeError("Only 0 or 1 is allowed")
            
            
