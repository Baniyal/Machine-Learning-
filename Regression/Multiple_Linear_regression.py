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

#predicting the test set result
y_pred = regressor.predict(X_test)
