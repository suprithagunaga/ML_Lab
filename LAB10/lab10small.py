import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centers = pca.transform(kmeans.cluster_centers_)
df = pd.DataFrame(X_pca, columns=['PC1','PC2'])
df['Cluster'] = clusters
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1')
plt.title('K-Means Clustering of Breast Cancer Dataset')
plt.show()
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1')
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, marker='x', label='Centroids')
plt.title('K-Means Clustering with Centroids')
plt.legend()
plt.show()
