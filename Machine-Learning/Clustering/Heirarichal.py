import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]] #selecting annual income and spending score

# using dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method = "ward"))
#ward method try to minimize the variance between each cluster
plt.title("dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidian distance")
plt.show()
# then find the largest distance in the graph we can get without cossing any horizontal line

# so here by setting a threshold we can see that there are 5 optimum clusters for
# this business problem


#fitting the heirarchichal clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5 , affinity = "euclidean" , linkage= "ward")
y_hc = hc.fit_predict(X)

#visualization of the clusters

plt.figure(figsize=(10, 7))  
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c= hc.labels_)
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')