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

path = "."
os.chdir(path)

# os.chdir(r"C:\Users\chris\Desktop\TATA\Penn State University\FALL 2019\DAAN 888\Project")
elnino = pd.read_csv('tao-all2.dat', header = None, delimiter=' ')
oni = pd.read_csv('ONI.csv', header = 0)
elnino.columns=['Observation', 'Year', 'Month', 'Day', 'Date', 'Latitude',
                'Longitude', 'Zonal Winds', 'Meridional Winds', 'Humidity', 'Air Temp', 'Sea Surface Temp']
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

## Fill NaNs with annual column means
# subsetting dataset by year
el_nino80 = elnino[elnino.Year == 80]
el_nino81 = elnino[elnino.Year == 81]
el_nino82 = elnino[elnino.Year == 82]
el_nino83 = elnino[elnino.Year == 83]
el_nino84 = elnino[elnino.Year == 84]
el_nino85 = elnino[elnino.Year == 85]      
el_nino86 = elnino[elnino.Year == 86]
el_nino87 = elnino[elnino.Year == 87]
el_nino88 = elnino[elnino.Year == 88]
el_nino89 = elnino[elnino.Year == 89]
el_nino90 = elnino[elnino.Year == 90]
el_nino91 = elnino[elnino.Year == 91]
el_nino92 = elnino[elnino.Year == 92]
el_nino93 = elnino[elnino.Year == 93]
el_nino94 = elnino[elnino.Year == 94]
el_nino95 = elnino[elnino.Year == 95]
el_nino96 = elnino[elnino.Year == 96]
el_nino97 = elnino[elnino.Year == 97]
el_nino98 = elnino[elnino.Year == 98]
# fill in NaN with column median for each year
el_nino80.fillna(el_nino80.median(), inplace=True)
el_nino81.fillna(el_nino81.median(), inplace=True)
el_nino82.fillna(el_nino82.median(), inplace=True)
el_nino83.fillna(el_nino83.median(), inplace=True)
el_nino84.fillna(el_nino84.median(), inplace=True)
el_nino85.fillna(el_nino85.median(), inplace=True)
el_nino86.fillna(el_nino86.median(), inplace=True)
el_nino87.fillna(el_nino87.median(), inplace=True)
el_nino88.fillna(el_nino88.median(), inplace=True)
el_nino89.fillna(el_nino89.median(), inplace=True)
el_nino90.fillna(el_nino90.median(), inplace=True)
el_nino91.fillna(el_nino91.median(), inplace=True)
el_nino92.fillna(el_nino92.median(), inplace=True)
el_nino93.fillna(el_nino93.median(), inplace=True)
el_nino94.fillna(el_nino94.median(), inplace=True)
el_nino95.fillna(el_nino95.median(), inplace=True)
el_nino96.fillna(el_nino96.median(), inplace=True)
el_nino97.fillna(el_nino97.median(), inplace=True)
el_nino98.fillna(el_nino98.median(), inplace=True)
# join the yearly dataframes into a single dataframe
elnino_1 = pd.concat([el_nino80, el_nino81, el_nino82, el_nino83, el_nino84,
                      el_nino85, el_nino86, el_nino87, el_nino88, el_nino89,
                      el_nino90, el_nino91, el_nino92, el_nino93, el_nino94,
                      el_nino95, el_nino96, el_nino97, el_nino98])
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
##
## Pre process ONI file
# rename columns
oni.columns=['Month','Year','Total','Anom']
print(oni.shape, "\n \n")
print(oni.dtypes, "\n \n")
# change name to month in digit form
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
# drop rows of data containing the following years
oni=oni[oni.Year != (1950)]
oni=oni[oni.Year != (1951)]
oni=oni[oni.Year != (1952)]
oni=oni[oni.Year != (1953)]
oni=oni[oni.Year != (1954)]
oni=oni[oni.Year != (1955)]
oni=oni[oni.Year != (1956)]
oni=oni[oni.Year != (1957)]
oni=oni[oni.Year != (1958)]
oni=oni[oni.Year != (1959)]
oni=oni[oni.Year != (1960)]
oni=oni[oni.Year != (1961)]
oni=oni[oni.Year != (1962)]
oni=oni[oni.Year != (1963)]
oni=oni[oni.Year != (1964)]
oni=oni[oni.Year != (1965)]
oni=oni[oni.Year != (1966)]
oni=oni[oni.Year != (1967)]
oni=oni[oni.Year != (1968)]
oni=oni[oni.Year != (1969)]
oni=oni[oni.Year != (1970)]
oni=oni[oni.Year != (1971)]
oni=oni[oni.Year != (1972)]
oni=oni[oni.Year != (1973)]
oni=oni[oni.Year != (1974)]
oni=oni[oni.Year != (1975)]
oni=oni[oni.Year != (1976)]
oni=oni[oni.Year != (1977)]
oni=oni[oni.Year != (1978)]
oni=oni[oni.Year != (1979)]
oni=oni[oni.Year != (1999)]
oni=oni[oni.Year != (2000)]
oni=oni[oni.Year != (2001)]
oni=oni[oni.Year != (2002)]
oni=oni[oni.Year != (2003)]
oni=oni[oni.Year != (2004)]
oni=oni[oni.Year != (2005)]
oni=oni[oni.Year != (2006)]
oni=oni[oni.Year != (2007)]
oni=oni[oni.Year != (2008)]
oni=oni[oni.Year != (2009)]
oni=oni[oni.Year != (2010)]
oni=oni[oni.Year != (2011)]
oni=oni[oni.Year != (2012)]
oni=oni[oni.Year != (2013)]
oni=oni[oni.Year != (2014)]
oni=oni[oni.Year != (2015)]
oni=oni[oni.Year != (2016)]
oni=oni[oni.Year != (2017)]
oni=oni[oni.Year != (2018)]
oni=oni[oni.Year != (2019)]
# replace years 4 digits to years 2 digits
oni=oni.replace(1980,80)
oni=oni.replace(1981,81)
oni=oni.replace(1982,82)
oni=oni.replace(1983,83)
oni=oni.replace(1984,84)
oni=oni.replace(1985,85)
oni=oni.replace(1986,86)
oni=oni.replace(1987,87)
oni=oni.replace(1988,88)
oni=oni.replace(1989,89)
oni=oni.replace(1990,90)
oni=oni.replace(1991,91)
oni=oni.replace(1992,92)
oni=oni.replace(1993,93)
oni=oni.replace(1994,94)
oni=oni.replace(1995,95)
oni=oni.replace(1996,96)
oni=oni.replace(1997,97)
oni=oni.replace(1998,98)
print(oni.head(200), "\n \n \n \n")
print(oni.describe())
#convert categorical variables to numeric
oni['Month']=pd.to_numeric(oni['Month'], errors='coerce')
print(oni.dtypes)
## joint the elnino and oni data files
# create left join
oni_1=oni.merge(elnino_1, on=['Year','Month'], how='left')
# display missing values
print("The total number of null/missing values before making any changes is: ")
print(oni_1.isnull().sum(), "\n")
print("So, there is :",oni_1.isnull().values.sum(), " values missing", "\n \n \n \n \n")
# drop na values for the 'Observation' variable (artifacts from the join)
oni_1.dropna(subset=['Observation'],inplace=True)

# display missing values
print("The total number of null/missing values after replacing nan values by the median of each column is: ")
print(oni_1.isnull().sum(), "\n")
print("So, there is :",oni_1.isnull().values.sum(), " value missing", "\n \n \n \n \n")

# Create target variable called 'weather_class'
#oni_1["weather_class"] = np.nan
oni_1['weather_class'] = np.where(oni_1['Anom'] <= -0.5, -1, np.where((oni_1['Anom'] > -0.5) & (oni_1['Anom'] < 0.5),0,1))
# change the 'weather_class' variable to categorical
oni_1['weather_class']=oni_1['weather_class'].astype('category')
oni_1.dtypes
print("The columns of the data set are now:", "\n")
print(oni_1.columns, "\n \n")
print("oni_1 data set: ", "\n")
print(oni_1.head(10), "\n \n")
print("The statistical summary of the Oni_1 data set is:", "\n")
print(oni_1.describe(), "\n \n")
print("The number of data per year is: ", "\n")
print(oni_1.groupby(['Year']).size(), "\n \n")
print("The number of data per Year for weather_class is: " , "\n")
print(oni_1.groupby(['Year','Month','weather_class']).size())
# export to csv
oni_1.to_csv('oni_1.csv')

#New correlation matrix
plt.matshow(oni_1[['Anom', 'Observation', 'Day', 'Date','Latitude', 'Longitude', 
                   'Zonal Winds', 'Meridional Winds', 'Humidity','Air Temp', 'Sea Surface Temp', 
                   'weather_class']].corr(), fignum=2)
plt.title('Correlation Matrix')
plt.colorbar()
plt.gca().xaxis.tick_bottom()
plt.xticks(range(13), list(oni_1.columns),rotation='vertical')
plt.yticks(range(13), list(oni_1.columns))
# print correlation values
oni_1.corr()

## subset the dataset to only years 89-98 to eliminate 'Humidity' missing values
# call the new dataset oni_2
oni89 = oni_1[oni_1.Year == 89]
oni90 = oni_1[oni_1.Year == 90]
oni91 = oni_1[oni_1.Year == 91]
oni92 = oni_1[oni_1.Year == 92]
oni93 = oni_1[oni_1.Year == 93]
oni94 = oni_1[oni_1.Year == 94]
oni95 = oni_1[oni_1.Year == 95]
oni96 = oni_1[oni_1.Year == 96]
oni97 = oni_1[oni_1.Year == 97]
oni98 = oni_1[oni_1.Year == 98]
oni_2 = pd.concat([ oni89,oni90,oni91,oni92,oni93,oni94,oni95,oni96,oni97,oni98])
# check shape and missing values
print(oni_2.isnull().sum(), "\n")
oni_2.shape
# Create a scatterplot matrix for new data consisting of columns [Zonal Winds,
# Meridional Winds, Humidity, Air Temp, Sea Surface Temp]
##  oni_3 is created to look at the independent variables of interest
oni_3 = oni_2[['Zonal Winds','Meridional Winds','Humidity', 'Air Temp','Sea Surface Temp']]
plt.style.use('classic')
pd.plotting.scatter_matrix(oni_3, s = 80, diagonal = 'kde')
# create boxplots for new data consisting of columns [Zonal Winds,
# Meridional Winds, Humidity, Air Temp, Sea Surface Temp]
color = dict(boxes='DarkGreen', whiskers='DarkOrange',
             medians='DarkBlue', caps='Gray')
oni_3.plot.box(color=color, sym='r+')
oni_3.boxplot()
plt.title("Boxplots for the variables Zonal Winds, Meridional Winds, Humidity, Air Temp and Sea Surface Temp")
# plot correlation matrix
names = ['Zonal Winds','Meridional Winds','Humidity', 'Air Temp','Sea Surface Temp']
correlations =oni_3.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,5,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.title("Correlation Matrix for oni-3")
plt.show()
# show correlation values
oni_3.corr()
oni_3.to_csv('oni_3.csv')
# create a new dataframe called oni_4 that includes the 4 variables of interest
# note: Air Temp not used because of the hige correlation with Sea Surface Temp
oni_4 = oni_2[['Zonal Winds','Meridional Winds','Humidity','Sea Surface Temp',]]
# translate Zonal Winds by + 13 to eliminate negative values for log transformation
oni_4['Zonal Winds']=oni_4['Zonal Winds'] + 13
# translate Meridional Winds by + 12 to eliminate negative values for log transformation
oni_4['Meridional Winds']=oni_4['Meridional Winds']+12
# Check to see if the min values > 0
oni_4.describe()
# log transformation of input variables creating new dataframe called oni_5
oni_5 = np.log(oni_4)
oni_5.describe()
oni_5.shape
# append the target variable weather_class to oni_5
oni_5['weather_class']= oni_2['weather_class']
print(oni_5.head(10), "\n \n")
print(oni_5.columns, "\n \n")
oni_5.dtypes
# export to csv
oni_5.to_csv('oni_5.csv')

# Prepare for modeling
from sklearn.model_selection import train_test_split
from sklearn import metrics
# define x and y variables
x=oni_5.iloc[:, 0 :4]
print(x.columns, "\n \n")
x.shape
y=oni_5.weather_class
y.shape
# split the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=34)
# Decision Tree Modeling
from sklearn import tree
DT = tree.DecisionTreeClassifier(max_depth=10, min_samples_split = 5)
DT.fit(x_train, y_train)
DT_pred=DT.predict(x_test)
# Decision Tree Model Evaluation
metrics.accuracy_score(DT_pred, y_test)
print(metrics.classification_report(y_test, DT_pred))
# Feature Importance
pd.DataFrame({'variable':oni_5.columns[:4],
              'importance':DT.feature_importances_})
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, DT_pred)
cm













