
#dataset utilized is from Kaggle
#link to dataset: https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/code


import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.cluster import KMeans 
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv(r'/Users/anuheaparker/Desktop/ml/clustering/apple_quality.csv')
print(df.head())
#print(df.info())
#print(df.describe())

#print(df.isnull().sum())
#print(df.duplicated().sum)

#remove the missing value 
df.dropna(inplace=True)
df.drop(columns=["A_id"], inplace=True)

df["Acidity"] = df["Acidity"].astype(float)
df["Quality"] = df["Quality"].map({"good":1, "bad":0})

#print(df.head())

#separating the target column out
float_columns = [x for x in df.columns if x not in ['Quality']]

# The correlation matrix
corr_mat = df[float_columns].corr()

# Strip out the diagonal values for the next step
for x in range(len(float_columns)):
    corr_mat.iloc[x,x] = 0.0
    
print(corr_mat)
print(corr_mat.abs().idxmax())


skew_columns = (df[float_columns]
                .skew()
                .sort_values(ascending=False))

skew_columns = skew_columns.loc[skew_columns > 0.75]
print(skew_columns)

sc = StandardScaler()
df[float_columns] = sc.fit_transform(df[float_columns])

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
sns.histplot(x=df['Size'])
plt.title('histplot of Size')

plt.subplot(2, 2, 2)
sns.histplot(x=df['Sweetness'])
plt.title('histplot of Sweetness')

plt.subplot(2, 2, 3)
sns.histplot(x=df['Crunchiness'])
plt.title('histplot of Crunchiness')

plt.subplot(2, 2, 4)
sns.histplot(x=df['Ripeness'])
plt.title('histplot of Ripeness')

plt.tight_layout()
plt.show()


#current version of matplotlib breaks seaborn heatmap annotations
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


sns.pairplot(df, hue='Quality')
plt.show()








km = KMeans(n_clusters=2, random_state=42)
km = km.fit(df[float_columns])

df['kmeans'] = km.predict(df[float_columns])


print((df[['Quality','kmeans']]
.groupby(['kmeans','Quality'])
.size()
.to_frame()
.rename(columns={0:'number'})))



km_list = list()

for clust in range(1,21):
    km = KMeans(n_clusters=clust, random_state=42)
    km = km.fit(df[float_columns])
    
    km_list.append(pd.Series({'clusters': clust, 
                            'inertia': km.inertia_,
                            'model': km}))

plot_data = (pd.concat(km_list, axis=1)
            .T
            [['clusters', 'inertia']]
            .set_index('clusters'))


ax = plot_data.plot(marker='o', ls='-')
ax.set_xticks(range(0,21,2), labels=(range(0,21,2)))
ax.set_xlim(0,21)
ax.set(xlabel='Cluster', ylabel='Inertia')
plt.show()



ag = AgglomerativeClustering(n_clusters=2, linkage="ward", compute_full_tree=True)
ag = ag.fit(df[float_columns])
df['agglom'] = ag.fit_predict(df[float_columns])

print((df[['Quality','agglom']]
.groupby(['agglom','Quality'])
.size()
.to_frame()
.rename(columns={0:'number'})))


db = DBSCAN(eps=5, min_samples=20)
db = db.fit(df[float_columns])
df['dbscan'] = db.fit_predict(df[float_columns])
print(f'DBSCAN found {len(set(db.labels_) - set([-1]))} clusters and {(db.labels_ == -1).sum()} points of noise.')


print((df[['Quality','dbscan']]
.groupby(['dbscan','Quality'])
.size()
.to_frame()
.rename(columns={0:'number'})))



"""

#Agg Clustering 
print((df[['Quality','agglom','kmeans']]
.groupby(['Quality','agglom'])
.size()
.to_frame()
.rename(columns={0:'number'})))

#Compare to KMeans results 
print((df[['Quality','agglom','kmeans']]
.groupby(['Quality','kmeans'])
.size()
.to_frame()
.rename(columns={0:'number'})))

#Compare results 
print((data[['Quality','agglom','kmeans']]
.groupby(['Quality','agglom','kmeans'])
.size()
.to_frame()
.rename(columns={0:'number'})))




"""





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