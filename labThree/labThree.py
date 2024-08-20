import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load CSV data function
def load_csv(filepath):
    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skipping the header
            values = line.strip().split(',')
            # Handle mixed data types: convert only numeric fields to float
            row = []
            for v in values:
                try:
                    # Try to convert to float
                    row.append(float(v))
                except ValueError:
                    # If conversion fails, keep as a string
                    row.append(v)
            data.append(row)
    return data

# Load the rent_vs_inc dataset
data = load_csv('/home/aditya/Documents/pythonLabZeroToFour/labThree/inc_vs_rent.csv')

# Inspect the first few rows of the data
for row in data[:5]:
    print(row)

# Extract rent and income values
rent = [row[0] for row in data]
income = [row[1] for row in data]

# Create scatter plot
plt.scatter(rent, income)
plt.xlabel('Rent')
plt.ylabel('Income')
plt.title('Rent vs Income Scatter Plot')
plt.show()

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
