import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Compute centroids for each cluster
centroids = data.groupby('cluster')[['x', 'y']].mean().values

# Cluster the centroids into 5 compact strata
agglo = AgglomerativeClustering(n_clusters=5, linkage='average')
strata_labels = agglo.fit_predict(centroids)

# Map cluster index to stratum
cluster_ids = data['cluster'].unique()
clusters_to_stratum = {cluster: stratum for cluster, stratum in zip(cluster_ids, strata_labels)}

# Assign stratum to each data point
data['stratum'] = data['cluster'].map(clusters_to_stratum)
