# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, -1].values#-1 means you take only the last  column

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#the first time around the SVR will give shit result if we 
#dont use feature scaling,here svr library doesnt include feature scaling on its own

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = np.squeeze(sc_Y.fit_transform(Y.reshape(-1, 1)))
#in the previous line we do this because fit_transform method expects a two dimensional 
#array as its input
# Fitting the Regression Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")
#rbf signifies the gaussian kernel
regressor.fit(X,Y)
# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
#since we scaled our arrays we also need to scale 6.5
print(y_pred) # this will give scaled value of y
#but in order to convert it into readable result
#so we need inverse transform method to get the expected value
y_pred = sc_Y.inverse_transform(y_pred)
print(y_pred)
# Visualising the Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

"""
# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""