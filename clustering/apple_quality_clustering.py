
#dataset utilized is from Kaggle
#link to dataset: https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/code




import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv(r'/Users/anuheaparker/Desktop/ml/clustering/apple_quality.csv')
print(df.head())
print(df.describe())

df.isnull().sum()
print(df.duplicated().sum)
















"""

# Normalize numerical features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Model training
k_values = range(2, 10)
inertia = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow curve to determine optimal K
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Based on the elbow curve, choose the optimal K and train the final model
optimal_k = 5
final_model = KMeans(n_clusters=optimal_k, random_state=42)
final_model.fit(df_scaled)

# Add cluster labels to the original dataframe
df['cluster'] = final_model.labels_

# Visualize the clusters (for 2D data)
plt.scatter(df['annual_income'], df['spending_score'], c=df['cluster'], cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()

# Explore cluster characteristics (e.g., demographic profile, spending behavior)
cluster_stats = df.groupby('cluster').mean()
print(cluster_stats)


"""