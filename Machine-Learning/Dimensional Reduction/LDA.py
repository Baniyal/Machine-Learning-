#using LDA to extract p<= n new independent variables that seperate
# the most the classes of the dependent variable
#LDA is supervised learning algorithm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv("Wine.csv")
# the three customer segment corresponds to the wine type that
# the customer segment like which kind of wine
X = dataset.iloc[:,0:13].values
Y = dataset.iloc[:,13].values
#so we have to predict the new kind of wine should  be
#catered to which kind of customer segment (reccomend to to them)


#we apply PCA to reduce the number of independent variables 

#splitting the dataset into the training and test set

from sklearn.model_selection import train_test_split
X_train , X_test ,Y_train ,Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 0 )


#feature scaling must be applied while doing dimensionality reduction
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)



#-------------------------------------Apply LDA after preprocessing data-----------------
#applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2) 
#number of principal components
#here we only want to take two principal compoenets to make it
# able to visualize
X_train = lda.fit_transform(X_train,Y_train)
#here since lda is a supervised learning algorithm,
# fit function will have two inputs,both the dependent and independent variables
X_test = lda.fit_transform(X_test,Y_test)


#this is the th step which tells how much a dependent variable
# will give the variance 
#this will contain %age of the variance explained by each of
# the principal components that we extracted here

#--------------------here we will take two components even though
#--------------------that doesnt make it to like 95% confidence level
#--------------------but for the sake of visualization we will do that


# fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,Y_train)

#Predicting the test results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(Y_test,y_pred)
#here we get 100 percent accuracy checking from confusion matrix
#since there are three classes we wont have a 2x2 confusion matrix
# here we have 3x3 confusion matrix 
# the diagonal contain the correctly predicted values
# refer to the video step 3- 4:00









# Visualizing the training set results
# the goal is to make the classifiers such that we can divide the set into
# two classes(prediction regions ),those who will buy the SUV and those who won't
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('green', 'yellow','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(("white", 'black',"grey"))(i), label = j)
plt.title('Logistic Regression using PCM(Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green',"blue")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green',"blue"))(i), label = j)
plt.title('Logistic Regression using PCM (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
