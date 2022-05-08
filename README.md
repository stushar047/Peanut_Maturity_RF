# Peanut_Maturity_Classification

## System Setup

Run the following lines to set up the system 

* Create a new environment
  * conda create --name Peanut
* Activate Peanut Maturity environment
  * conda activate Peanut
* Install All Libraries
  * pip install envi
  * pip install opencv-python
  * conda install matplotlib
  * conda install scikit-learn
  * conda install pandas
  * conda install pillow
  * install git [installation instruction for windows](https://github.com/git-guides/install-git)
* Change the directory where you want to download all the files
  * cd Directory (Example: cd C:/Users/tushar/Documents) 
* Download all the files
  * git clone https://github.com/stushar047/Peanut_Maturity_Classification.git  

## Run the code

* For running the code, always make sure that you are in the write environment and write directory 
  * conda activate Peanut
  * cd Peanut_Maturity_Classification
  * download the files
* Run the code
  * python datacreation.py filename_before_space_portion (python datacreation.py Example: 3A_1-15_Side1_20160914_0001_cnbh)
## Collect all the files required
There will be two types of file that needs to be collected and uploaded in a google drive
1. Image file<br>
Image files are RGB_image_{{filename}.jpg, Ostu_thresholded_image_{filename}.jpg, Morphologically_processed_image{filename}.jpg and Peanut_identification_{filename}.jpg 
2. csv file<br> 
Spectral.csv, Spatial_spectral.csv

## Check the plots to make sure everything is right
* Run this code
  * python Plots.py
* Match the Plots with
  * Check the plots with figure 4 of this [paper](https://www.sciencedirect.com/science/article/pii/S1537511019308621)    


