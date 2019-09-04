# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values
#in order to see x and y(the numpy arrays you can do so by converting it into DATAFRANE)
# Splitting the dataset into the Training set and Test set

#make the column of location into the categorical data
#---------------------CATEGORICAL DATA--------------------


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#one hot encoder cannot be used directly without the label encoder
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#--------------------AVOIDING DUMMY VARIABLE TRAP
X = X[:,1:]
#making sure dataset doesnt contain redundant dependencies

#-----------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


#------------------BACKWARD IMPLEMENTATION---------------------
#what if there are some variables are statistically significant
# we need to find such variables
import statsmodels.api as sm
# to make statsmodel to understand
# y = b0*x0 + b1*x1 ----- (where x0 = 1)
#so we add a column of all ones
X = np.append(arr = np.ones((50,1)).astype(int) , values = X , axis = 1)
#----------------------Starting backward elimination------------
#crerating a new matrix having optimal features only which have higher
#statistical impact
X_opt = X[:,[0,3,4,5]] 
# creatinf new regressor object from class
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
#OLS doesnt add intercept on its own
#look for predictor which has highest p value
print(regressor_OLS.summary())
#SUMMARY VERY IMPOERTANT FUNCTION
#look for highest value 
#if p value is greater than significant value we remove it
#so we remove x1 and x2 so we make changes in life 52
#keep doing in this in loop untill all variabkes have p value < significant level
X_opt = X[:,[0,1,3,4,5]] 
X_opt = X[:,[0,3,4,5]] 
X_opt = X[:,[0,3,5]] 
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
print(regressor_OLS.summary())
# we need to remove even 3rd index column because p value slightly greater than Significance value
X_opt = X[:,[0,3]] 
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
print(regressor_OLS.summary())
#hence only one significant independent variable