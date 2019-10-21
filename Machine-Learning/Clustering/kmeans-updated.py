#importing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the mall data set with pandas
dataset = pd.read_csv("Mall_Customers.csv")

#clustering problem when we dont know the answers and are looking ath the data to 
#provide us with insights
X = dataset.iloc[:,[3,4]].values #annual income column and spending score

#now we are going to find out the number of optiimal number of clusters

#using the elbow methode(graph vaaala) to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = [ ]
for i in range(1 ,11):
    kmeans = KMeans(n_clusters= i , init = "k-means++",max_iter = 300 , n_init = 10)
    kmeans.fit(X)
    #fitting means into the data
    #wcss is also known as inertia
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11) , wcss )
plt.title("the elbow method")
plt.xlabel("the number of clusters")
plt.ylabel("WCSS")
plt.show()
#from graph we can tell that at 5 number of clusters there was a huge dip and hence
#it is the optimmal value

#now we apply kmeans method for  5 number of cluster
kmeans = KMeans(n_clusters = 5 , init = "k-means++" , max_iter = 300 , n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#-------------------------Silhoutte analysis(can also be used with other clustering techniques)------------
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_kmeans)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X , y_kmeans , metric = "euclidean")
#visualization of silhouette analysis
y_ax_lower , y_ax_upper = 0 , 0 
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_kmeans == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    #color = cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower , y_ax_upper) , 
             c_silhouette_vals , 
             height = 1.0 ,
             edgecolor = "none" , 
             color ="Red" )
    yticks.append((y_ax_lower + y_ax_upper) / 2.0 )
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg , 
            color = "White" ,
            linestyle = "--"
           )
plt.yticks(yticks , cluster_labels + 1 )
plt.ylabel("Cluster")
plt.xlabel("Silhoutte Coefficient")
plt.show()

#visualizing the clusters
# Visualising the cluster
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()