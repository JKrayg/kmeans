import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Apply k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Convert to DataFrame for easier visualization
iris_df = pd.DataFrame(X, columns=feature_names)
iris_df['cluster'] = labels

# Plot the scatterplot
plt.figure(figsize=(8, 6))

# Scatter plot of two features (sepal length vs. sepal width)
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='cluster', palette='viridis', style='cluster', markers=["o", "s", "D"])

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')

plt.title('Clustering Results for Iris Dataset (KMeans)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()
