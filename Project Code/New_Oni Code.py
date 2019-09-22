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
##
##
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
print(elnino_1.head(13), "\n \n \n \n")
##
##
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
# create left join
oni_1=oni.merge(elnino_1, on=['Year','Month'], how='left')
print(oni_1.head(15))
oni_1.to_csv(r'C:\Users\chris\Desktop\TATA\Penn State University\FALL 2019\DAAN 888\Project\New_Oni_missing_values.csv')
# display missing values
print("The total number of null/missing values before making any changes is: ")
print(oni_1.isnull().sum(), "\n")
print("So, there is :",oni_1.isnull().values.sum(), " values missing", "\n \n \n \n \n")
## get name of columns in dataset
print("The name of the columns left in dataset are:")
print(oni_1.columns, "\n")
# drop na values
oni_1.fillna(oni_1.median(), inplace=True)
# display missing values
print("The total number of null/missing values after replacing nan values by the median of each column is: ")
print(oni_1.isnull().sum(), "\n")
print("So, there is :",oni_1.isnull().values.sum(), " value missing", "\n \n \n \n \n")
## get number of rows and column in dataset
print("The dataset file is composed of the following number of rows and columns (rows, columns): ")
print(oni_1.shape, "\n")
print(oni_1.head(10), "\n \n \n \n")
oni_1.to_csv(r'C:\Users\chris\Desktop\TATA\Penn State University\FALL 2019\DAAN 888\Project\New_Oni_filled_median.csv')
oni = pd.read_csv("New_Oni_filled_median.csv")
oni.head()
oni["Year_Type"] = np.nan
oni.columns
oni['Year_Type'] = np.where(oni['Anom'] <= -0.5, -1, np.where((oni['Anom'] > -0.5) & (oni['Anom'] < 0.5),0,1))