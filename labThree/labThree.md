### Explanation of Lab 3: Classification with K-means Clustering and Hyper-parameter Optimization using Grid Search

This lab focuses on understanding and implementing the K-means clustering algorithm from scratch, followed by optimizing the number of clusters using grid search and silhouette scores. You will also classify new data points using the trained K-means model.

---

### **Task 1: Classification with K-means**

#### **Task 1.1: Load the Dataset**

You will first load the dataset, `rent_vs_inc.csv`, in the same way as in previous labs, using custom file I/O operations (without using libraries like pandas for the loading process).

#### Code for Task 1.1: Loading the Dataset

```python
def load_csv(filepath):
    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skipping header
            values = line.strip().split(',')
            data.append([float(v) for v in values])
    return data

# Load the rent_vs_inc dataset
data = load_csv('rent_vs_inc.csv')

# Inspect the first few rows of the data
for row in data[:5]:
    print(row)
```

The dataset contains columns such as average rent and income per region. After loading the dataset, you should inspect the values to ensure they were read correctly.

#### **Task 1.2: Scatter Plot**

To visualize the dataset, you will plot the points (rent vs income) on a scatter plot using Matplotlib.

#### Code for Task 1.2: Scatter Plot

```python
import matplotlib.pyplot as plt

# Extract rent and income values
rent = [row[0] for row in data]
income = [row[1] for row in data]

# Create scatter plot
plt.scatter(rent, income)
plt.xlabel('Rent')
plt.ylabel('Income')
plt.title('Rent vs Income Scatter Plot')
plt.show()
```

#### **Task 1.3: Implement K-means Clustering from Scratch**

K-means clustering involves:
1. Initializing the centroids randomly.
2. Assigning each point to the nearest centroid (using Euclidean distance).
3. Recalculating the centroids as the mean of the points assigned to each cluster.
4. Repeating the process until the centroids do not change significantly.

#### Code for Task 1.3: K-means Clustering Algorithm

```python
import numpy as np

# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

# K-means clustering function
def kmeans(data, k, iterations=10):
    # Randomly initialize centroids
    centroids = data[np.random.choice(len(data), k, replace=False)]

    for _ in range(iterations):
        # Assign points to the closest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid_idx = np.argmin(distances)
            clusters[closest_centroid_idx].append(point)

        # Recompute centroids
        new_centroids = [np.mean(cluster, axis=0) if len(cluster) > 0 else centroids[i] for i, cluster in enumerate(clusters)]

        # Check for convergence (centroids don't change)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters

# Convert data to NumPy array
data_np = np.array(data)

# Apply K-means with 3 clusters
centroids, clusters = kmeans(data_np, 3)

# Plot the clusters with different colors
colors = ['r', 'g', 'b']
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i])

# Plot the centroids
centroids = np.array(centroids)
plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='x', label='Centroids')

plt.xlabel('Rent')
plt.ylabel('Income')
plt.title('K-means Clustering (3 Clusters)')
plt.show()
```

This code implements the K-means clustering algorithm and visualizes the clusters along with their centroids.

---

### **Task 2: Hyper-parameter Optimization**

In this task, you will use grid search to find the optimal number of clusters for K-means by calculating the **silhouette score** for each value of k (number of clusters).

#### **Silhouette Score**

The silhouette score measures how similar a data point is to its own cluster (intra-cluster distance) compared to other clusters (inter-cluster distance). A higher silhouette score indicates better-defined clusters.

#### Code for Task 2: Silhouette Score and Grid Search

```python
from sklearn.metrics import silhouette_score

# Silhouette score function
def silhouette_coefficient(data, centroids, clusters):
    labels = np.zeros(len(data))
    for cluster_idx, cluster in enumerate(clusters):
        for point in cluster:
            point_idx = np.where((data == point).all(axis=1))[0][0]
            labels[point_idx] = cluster_idx
    return silhouette_score(data, labels)

# Grid search for the optimal number of clusters (1-10)
def grid_search(data, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        centroids, clusters = kmeans(data, k)
        score = silhouette_coefficient(data, centroids, clusters)
        silhouette_scores.append(score)
    return silhouette_scores

# Perform grid search and plot the silhouette scores
silhouette_scores = grid_search(data_np, 10)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Cluster Sizes')
plt.show()

# Find the optimal number of clusters
optimal_k = np.argmax(silhouette_scores) + 2  # Adding 2 because the range starts at 2
print(f'Optimal number of clusters: {optimal_k}')
```

This code performs a grid search to find the optimal number of clusters by calculating the silhouette score for values of k from 2 to 10. The silhouette scores are plotted to help visualize the optimal cluster size.

---

#### **Task 2.2: Scatter Plot with Optimal Number of Clusters**

Once the optimal number of clusters is found, you will plot the clusters again with the new number of clusters.

```python
# Apply K-means with the optimal number of clusters
optimal_centroids, optimal_clusters = kmeans(data_np, optimal_k)

# Plot the clusters with different colors for the optimal k
for i, cluster in enumerate(optimal_clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i % len(colors)])

# Plot the centroids
plt.scatter(optimal_centroids[:, 0], optimal_centroids[:, 1], c='k', marker='x', label='Centroids')

plt.xlabel('Rent')
plt.ylabel('Income')
plt.title(f'K-means Clustering with {optimal_k} Clusters')
plt.show()
```

---

### **Task 2.3: Classify New Data Points**

Now, you will classify new points based on the optimal K-means model.

```python
# New points to classify
new_points = np.array([[1010, 320.12], [1258, 320], [980, 292.4]])

# Classify new points
for point in new_points:
    distances = [euclidean_distance(point, centroid) for centroid in optimal_centroids]
    closest_centroid_idx = np.argmin(distances)
    print(f'Point {point} belongs to cluster {closest_centroid_idx + 1}')

# Plot the new points with the clusters
for i, cluster in enumerate(optimal_clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i % len(colors)])

# Plot the new points
plt.scatter(new_points[:, 0], new_points[:, 1], c='purple', label='New Points')

# Plot the centroids
plt.scatter(optimal_centroids[:, 0], optimal_centroids[:, 1], c='k', marker='x', label='Centroids')

plt.xlabel('Rent')
plt.ylabel('Income')
plt.title('Classifying New Data Points')
plt.legend()
plt.show()
```

This code classifies the new data points by assigning them to the nearest cluster and plots them along with the existing clusters.

---

### **Optional Advanced Task: N-dimensional Grid Search Optimizer**

To implement an N-dimensional grid search for optimizing multiple hyperparameters, you can extend the grid search function by iterating over combinations of multiple hyperparameters and calculating the evaluation metric (such as silhouette score) for each combination.

---

### Summary:

- **Task 1**: You implemented K-means clustering from scratch and visualized the clusters.
- **Task 2**: You performed grid search to find the optimal number of clusters using silhouette scores and re-visualized the clusters.
- **Task 2.3**: You classified new data points based on the trained K-means model.

Let me know if you need further assistance with any part of the lab!