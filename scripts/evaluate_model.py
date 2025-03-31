

# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error,mean_squared_error , r2_score

     

#Loding the data
housing= pd.read_csv("/Users/sumanshrestha/Documents/AI Class Omdena/machine-learning-introduction-makaisuman/data/boston_housing.csv")
     

#Since CAT.MEDV is not required removing it
housing= housing.drop(columns=['CAT. MEDV'])
     

#checking the data information
housing.info()
     


###  Handle Outliers using IQR Method ###
Q1 = housing.quantile(0.25)
Q3 = housing.quantile(0.75)
IQR = Q3 - Q1
     

# Define outlier range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
     

# Remove outliers
housing = housing[~((housing < lower_bound) | (housing > upper_bound)).any(axis=1)]

     

# Converting all the data to float
# we need to convert the CHAS, RAD and TAX

housing["CHAS"] =housing["CHAS"].astype(float)  # for CHAS
housing["RAD"] =housing["RAD"].astype(float) # for RAD
housing["TAX"] =housing["TAX"].astype(float)  #For TAX
     

# Standarizing  the data
scaler = StandardScaler()
scaled_data= scaler.fit_transform(housing)
scaled_df = pd.DataFrame(scaled_data, columns=housing.columns)

     

# Split my data into the features (X) and target y
X = scaled_df.drop(columns = ['MEDV'])
y = scaled_df['MEDV']
     

# Split the data into the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
     

# Train the model
model = LinearRegression() # Innitialize the model
model.fit(X_train, y_train)
     


# Prediction
y_pred = model.predict(X_test)
     


print(LinearRegression)

# Evaluate with MAE and MSE
mae = mean_absolute_error(y_true=y_test,y_pred=y_pred) 
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True 
r2 = r2_score(y_test, y_pred)
print("LinearRegression")
print("MAE:",mae) 
print("MSE:",mse) 
print("R²:",r2)
     

#Using Ridge Regression (L2 Regularization)
ridgeModel = Ridge(alpha=1.0)
ridgeModel.fit(X_train, y_train)
ridge_pred = ridgeModel.predict(X_test)
     

# Evaluating Rigie Model
print("Ridge Regression")
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
print("MSE:",ridge_mse) 
print("R²:",ridge_r2) 
     


lassoModel = Lasso(alpha=0.1)
lassoModel.fit(X_train, y_train)
lasso_pred = lassoModel.predict(X_test)
     

# Evaluating the Lasso Regression Model
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)
print("Lasso Regression Model")
print("MSE:",lasso_mse) 
print("R²:",lasso_r2) 
     
