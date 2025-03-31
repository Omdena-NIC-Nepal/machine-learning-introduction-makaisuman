# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Loding the data
housing= pd.read_csv("/Users/sumanshrestha/Documents/AI Class Omdena/machine-learning-introduction-makaisuman/data/boston_housing.csv")

#checking the data information
housing.info()

#Since CAT.MEDV is not required removing it
housing= housing.drop(columns=['CAT. MEDV'])


#Check the missing values
housing.isnull().sum()

# Handeling the missing values and putting median as missing values
housing.fillna(housing.median(), inplace=True) 

housing.head()

### Handle Outliers using IQR Method ###
Q1 = housing.quantile(0.25)
Q3 = housing.quantile(0.75)
IQR = Q3 - Q1

# Define outlier range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
housing = housing[~((housing < lower_bound) | (housing > upper_bound)).any(axis=1)]

# Checking the data details and type before endoding
housing.info()

# Converting all the data to float Encoding
# we need to convert the CHAS, RAD and TAX

housing["CHAS"] =housing["CHAS"].astype(float)  # for CHAS
housing["RAD"] =housing["RAD"].astype(float) # for RAD
housing["TAX"] =housing["TAX"].astype(float)  #For TAX

# Checking the data details and type after endoding
housing.info()

# Standarizing  the data
scaler = StandardScaler()
scaled_data= scaler.fit_transform(housing)
scaled_df = pd.DataFrame(scaled_data, columns=housing.columns)


# Data after Standardization
scaled_df.head()

# Split my data into the features (X) and target y
X = scaled_df.drop(columns = ['MEDV'])
y = scaled_df['MEDV']

# Split the data into the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)