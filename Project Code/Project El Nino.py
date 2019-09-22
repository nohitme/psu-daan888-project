# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 08:33:51 2019

@author: chris
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

os.chdir(r"C:\Users\chris\Desktop\TATA\Penn State University\FALL 2019\DAAN 888\Project")
elnino = pd.read_csv('tao-all2.dat', header = None, delimiter=' ')
oni = pd.read_csv('ONI.csv', header = 0)
elnino.columns=['Observation', 'Year', 'Month', 'Day', 'Date', 'Latitude', 'Longitude', 'Zonal Winds', 'Meridional Winds', 'Humidity', 'Air Temp', 'Sea Surface Temp']
##
# Print statistical summary
print("Statistical summary for the 'Elnino' file: \n")
print(elnino.describe(), "\n \n")
# display types for variables
print("The variable types are as follow: \n")
print(elnino.dtypes, "\n")
## get name of columns in dataset
print("The name of the columns in dataset are: \n")
print(elnino.columns, "\n")
## get number of rows and column in dataset
print("The dataset file is composed of the following number of rows and columns (rows, columns): ")
print(elnino.shape, "\n \n")
# looking for null values
print("The total number of null/missing values before converting the variables is: ")
print(elnino.isnull().sum(), "\n")
print("So, there is :",elnino.isnull().values.sum(), " value missing", "\n \n \n \n \n")
##
#convert categorical variables to numeric
elnino['Zonal Winds']=pd.to_numeric(elnino['Zonal Winds'], errors='coerce')
elnino['Meridional Winds']=pd.to_numeric(elnino['Meridional Winds'], errors='coerce')
elnino['Humidity']=pd.to_numeric(elnino['Humidity'], errors='coerce')
elnino['Air Temp']=pd.to_numeric(elnino['Air Temp'], errors='coerce')
elnino['Sea Surface Temp']=pd.to_numeric(elnino['Sea Surface Temp'], errors='coerce')
# display data types
print("After converting the variables, the new Data types are: \n",elnino.dtypes, "\n")
# replace empty spaces with NAN values
elnino.replace(r'^\s*$', np.nan, regex=True)
print("The count of filled cells for each variable in the data set is: \n")
print(elnino.apply(lambda x: x.count(), axis=0), "\n \n")
print("The sum of filled cells in the data set is: ")
print(elnino.apply(lambda x: x.count(), axis=0).sum(), "\n \n")
# print missing values per year
elnino_Latitude = elnino['Latitude'].isnull().groupby(elnino['Year']).sum().astype(int).reset_index(name='Latitude')
elnino_Longitude = elnino['Longitude'].isnull().groupby(elnino['Year']).sum().astype(int).reset_index(name='Longitude')
elnino_Zonal_Winds = elnino['Zonal Winds'].isnull().groupby(elnino['Year']).sum().astype(int).reset_index(name='Zonal Winds')
elnino_Meridional_Winds = elnino['Meridional Winds'].isnull().groupby(elnino['Year']).sum().astype(int).reset_index(name='Meridional Winds')
elnino_Humidity = elnino['Humidity'].isnull().groupby(elnino['Year']).sum().astype(int).reset_index(name='Humidity')
elnino_Air_Temp = elnino['Air Temp'].isnull().groupby(elnino['Year']).sum().astype(int).reset_index(name='Air Temp')
elnino_Sea_Surface_Temp = elnino['Sea Surface Temp'].isnull().groupby(elnino['Year']).sum().astype(int).reset_index(name='Sea Surface Temp')
elnino_1=elnino_Latitude.join(elnino_Longitude.set_index('Year'),on='Year')
elnino_2=elnino_1.join(elnino_Zonal_Winds.set_index('Year'),on='Year')
elnino_3=elnino_2.join(elnino_Meridional_Winds.set_index('Year'),on='Year')
elnino_4=elnino_3.join(elnino_Humidity.set_index('Year'),on='Year')
elnino_5=elnino_4.join(elnino_Air_Temp.set_index('Year'),on='Year')
elnino_6=elnino_5.join(elnino_Sea_Surface_Temp.set_index('Year'),on='Year')
print("The total number of missing values per variable grouped by Year is: ","\n")
print(elnino_6,"\n \n")
##
# looking for null values
print("The total number of null/missing values for each variable is: ", "\n")
print(elnino.isnull().sum(), "\n")
print("So, there is :",elnino.isnull().values.sum(), " value missing", "\n \n \n \n \n")
print(elnino.head(10))
elnino_data_sum=elnino.apply(lambda x: x.count(), axis=0).sum()
elnino_missing_sum=elnino.isnull().values.sum()
# display percentage of missing data
print("The percentage of missing data is: ", "\n")
print(str(round((elnino_missing_sum/elnino_data_sum)*100,2)),"%" "\n \n \n \n")
# print correlation matrix
#plt.figure(figsize=(12,12))
plt.matshow(elnino.corr(), fignum=2)
plt.title('Correlation Matrix El Nino')
plt.colorbar()
plt.gca().xaxis.tick_bottom()
plt.xticks(range(12), list(elnino.columns),rotation='vertical')
plt.yticks(range(12), list(elnino.columns))
# print histograms
print(elnino.hist(column=['Latitude','Longitude', 'Zonal Winds', 'Meridional Winds', 'Humidity', 'Air Temp', 'Sea Surface Temp'], figsize=(10,10)),"\n \n \n")
##
# print Seaborn Joint Scatterplot
print(sns.jointplot(x="Air Temp", y="Sea Surface Temp", data=elnino, size=7, alpha=0.3),"\n \n")
print(sns.jointplot(x="Air Temp", y="Humidity", data=elnino, size=7, alpha=0.3),"\n \n \n")
##
###### If na values are dropped
## drop unecessary variables
#elnino_1= elnino.drop(['Observation','Year','Month','Day','Date'], axis=1)
### get name of columns in dataset
#print("The name of the columns in dataset are:")
#print(elnino_1.columns, "\n")
## drop na values
#elnino_1= elnino_1.dropna()
## display missing values
#print("The total number of null/missing values is: ")
#print(elnino_1.isnull().sum(), "\n")
#print("So, there is :",elnino_1.isnull().values.sum(), " value missing", "\n \n \n \n \n")
### get number of rows and column in dataset
#print("The dataset file is composed of the following number of rows and columns (rows, columns): ")
#print(elnino_1.shape, "\n")
## print Seaborn Matrix Correlation
##print(sns.heatmap(elnino_1.corr()), " \n \n \n")
###### If na values are imputed with mean values for each column
# drop unecessary variables
elnino_1= elnino.drop(['Observation','Day','Date'], axis=1)
## get name of columns in dataset
print("The name of the columns left in dataset are:")
print(elnino_1.columns, "\n")
# drop na values
elnino_1.fillna(elnino_1.median(), inplace=True)
# display missing values
print("The total number of null/missing values after replacing nan values by the median of each column is: ")
print(elnino_1.isnull().sum(), "\n")
print("So, there is :",elnino_1.isnull().values.sum(), " value missing", "\n \n \n \n \n")
## get number of rows and column in dataset
print("The dataset file is composed of the following number of rows and columns (rows, columns): ")
print(elnino_1.shape, "\n")
print(elnino_1.head(10), "\n \n \n \n")
## Detect outliers
#print("\n \n",elnino.boxplot(column=["Air Temp"]),"\n \n")
#print(elnino.boxplot(column=['Humidity']),"\n \n")
#print(elnino.boxplot(column=['Meridional Winds']),"\n \n")
#print(elnino.boxplot(column=['Zonal Winds']),"\n \n")
#print(elnino.boxplot(column=['Sea Surface Temp']))
## Pre process ONI file
oni.columns=['Month','Year','Total','Anom']
print(oni.shape)
print(oni.dtypes)
oni=oni.replace('DJF', '12')
oni=oni.replace('JFM', '1')
oni=oni.replace('FMA', '2')
oni=oni.replace('MAM', '3')
oni=oni.replace('AMJ', '4')
oni=oni.replace('MJJ', '5')
oni=oni.replace('JJA', '6')
oni=oni.replace('JAS', '7')
oni=oni.replace('ASO', '8')
oni=oni.replace('SON', '9')
oni=oni.replace('OND', '10')
oni=oni.replace('NDJ', '11')
print(oni)
#convert categorical variables to numeric
oni['Month']=pd.to_numeric(oni['Month'], errors='coerce')
print(oni.dtypes)
