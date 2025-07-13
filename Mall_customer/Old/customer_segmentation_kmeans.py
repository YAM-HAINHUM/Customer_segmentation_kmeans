# customer_segmentation_kmeans.py

# 📦 Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 📥 Load dataset
df = pd.read_csv("Mall_Customers.csv")

# 👀 Display basic info
print("First 5 rows:\n", df.head())

# 📊 Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 📈 Visualize the data
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue=None)
plt.title("Customer Distribution")
plt.show()

# 🔍 Use the Elbow Method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 📉 Plot the Elbow Graph
plt.figure(figsize=(8,6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# 💡 Choose optimal number of clusters (e.g., 5 based on elbow)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 🏷️ Add cluster labels to the original dataframe
df['Cluster'] = y_kmeans

# 🎨 Visualize the clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=X['Annual Income (k$)'], y=X['Spending Score (1-100)'],
                hue=y_kmeans, palette='bright')
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title='Cluster')
plt.show()

# 📋 Summary of clusters
print("\nCluster Centers:")
print(kmeans.cluster_centers_)
