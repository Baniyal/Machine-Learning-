# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
# Create your regressor here
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300 , random_state = 0) #n_estimatoes
# is the number of decision trees that  will be made whose average will be given as the prediction
regressor.fit(X,y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])
y_pred
""" #not going to use because its basically a collection of decision tree
# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""
# Visualising the Regression results (for higher resolution and smoother curve)
#by having a  mult iple steps in the whole graph because there are
#multiple decision trees which in turn represent more number of splits
# which is more number of steps 
#Althoug adding more number of trees doesnt mean that there will be more numberr of steps
#because the more the average of the different predictions made by the trees is
#converging to the same average ()
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random forest regresssion)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()