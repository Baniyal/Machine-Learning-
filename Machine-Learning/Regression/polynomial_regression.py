# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #we do 1:2 which is the same as 1: in thsi case,its so that we dont get an error if x is a matrix but not an array
Y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
#not going to do it here because the number of entries are very small
# and we want to make accurate assumptions so we use X as training set alone
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train ,Y_test = train_test_split(X , Y ,test_size = 0.2 ,random_state= 0 )
# Feature Scaling
#no need here as we need same library for linear regression
# and that library does feature scaling on their own
#------------Fitting Linear regression to the data set
from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(X,Y)

#---------Fitting Polynomial Regression to the data set
from sklearn.preprocessing import PolynomialFeatures
#this will transform X into X_poly
poly_reg = PolynomialFeatures(degree = 4) # this degree will create 2 columnns which are the polynomial term
X_poly = poly_reg.fit_transform(X)
#it adds additional polynomial values onto the X
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

X_test_trans = poly_reg.fit_transform(X_test)

Y_pred  = lin_reg2.predict(X_test_trans)

from sklearn.metrics import r2_score
r2 = r2_score(Y_test,Y_pred)
print(r2)

from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test , Y_pred) 
"""
Y_pred  = lin_reg2.predict(X_test)

from sklearn.metrics import r2_score
r2 = r2_score(Y_test,Y_pred)
print('{:.3f}'.format(r2))
"""
#this will combine X_poly will be fitted with the regressor

#------Visualizing the Linear Regression Results-------
plt.scatter(X,Y,color ="red")
plt.plot(X,lin_reg1.predict(X),color = "blue")
plt.title("Truth or Bluff(Linear regresion)")
plt.xlabel("position label")
plt.ylabel("salary")
plt.show()



#------Visualizing the Polynomial Regression Results
plt.scatter(X,Y,color = "red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="pink")
plt.title("Truth or Bluff using polynomial Regression")
plt.xlabel("position label")
plt.ylabel("salary")
plt.show()
"""
# as we increase the degreee in line 28 it gets better at predictions
#-------------IMPROVING PLOT
#since there were straight lines between the points in the
#polynomial regression
# we can haveprediction at a higher resolution to get the proper curve
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid)),1)
 #------Visualizing the Polynomial Regression Results
plt.scatter(X,Y,color = "red")
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color="blue")
plt.title("Truth or Bluff using polynomial Regression")
plt.xlabel("position label")
plt.ylabel("salary")
plt.show()
"""
#predicting a new result with linear regression
lin_reg1.predict([[6.4]])# always use [[]] because it expects a 2D array as an input]

#predicting a new result with polynomial linear regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))