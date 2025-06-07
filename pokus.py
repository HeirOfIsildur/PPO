import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import string

from plot_cluster import (
    plot_cluster_grid,
    plot_cluster_grid_highlight,
    plot_stratum_grid_highlight,
)

#Constants
seed = 42 #my UKČO
percentage_white_pixels = 0.05
N_clusters = 20
N_strata = 5



random.seed(seed)
np.random.seed(seed)

# Read the CSV file
data = pd.read_csv("soildata.csv", sep=";")
data = data[["x", "y", "Cu"]]
data["Cu"] = data["Cu"].str.replace(",", "", regex=False).astype(int)
True_total = data["Cu"].sum()
True_mean = data["Cu"].mean()
# Remove 5% of the rows at random (white pixels, we act like there are not existing at all so number of PSUs doesn't include white pixels)
data = data.sample(frac=1-percentage_white_pixels, random_state=seed).reset_index(drop=True)

kmeans = KMeans(n_clusters=N_clusters, random_state=seed)
data['cluster'] = kmeans.fit_predict(data[['x', 'y']])
print(data[['cluster']].value_counts())
plot_cluster_grid(data = data, plot_variable='cluster', title='Soil Data Clustering')

data["cluster"].value_counts()
clusters = data.groupby('cluster').apply(lambda df: df.index.tolist()).to_dict()

# Get unique clusters and shuffle them
unique_clusters = list(data['cluster'].unique())
np.random.shuffle(unique_clusters)

strata_labels = list(range(N_strata))  # [0, 1, 2, 3, 4]

# Map clusters to strata randomly
cluster_to_stratum = {}
clusters_per_stratum = len(unique_clusters) // N_strata
for i, cluster in enumerate(unique_clusters):
    stratum = strata_labels[i // clusters_per_stratum]
    cluster_to_stratum[cluster] = stratum

data['stratum'] = data['cluster'].map(cluster_to_stratum)
plot_cluster_grid(data=data, plot_variable='stratum', title='Soil Data Clustering (Strata)')
strata = {}
for cluster, stratum in cluster_to_stratum.items():
    strata.setdefault(stratum, []).append(cluster)

# #Modelling part:
# # A) Clusters selected with probabilities proportional to size, with replacement
events = np.arange(N_clusters)
probabilities = data['cluster'].value_counts(normalize=True).sort_index().values
n_psu_samples = 5
selected_clusters = np.random.choice(events, size=n_psu_samples, replace=True, p=probabilities)
print(selected_clusters)
plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters, title='Soil Data Clustering (Strata)', )

n_ssu_samples = 5
ssu_samples = {}
for i, cluster in enumerate(selected_clusters):
    ssu_samples_per_cluster = np.random.choice(clusters[cluster], size=n_ssu_samples, replace=False)
    ssu_samples[f"{cluster}_{i}"] = list(ssu_samples_per_cluster)

ssu_samples_plot = [item for sublist in ssu_samples.values() for item in sublist]
plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters, title='Soil Data Clustering (Strata)', selected_points=ssu_samples_plot )

A_calculations = {}
for cluster, samples in ssu_samples.items():
    cluster_without_underscore = int(cluster.split('_')[0])  # Extract the cluster number
    sample_cluster_observations = data.loc[samples, 'Cu']
    A_calculations[cluster] = {
        "total_cluster_size": len(clusters[cluster_without_underscore]),
        "sample_cluster_size": len(samples),
        "sample_mean": sample_cluster_observations.mean(),
        "sample_total": sample_cluster_observations.mean() * len(clusters[cluster_without_underscore]),
        "sample_variance": sample_cluster_observations.var(ddof=1),
    }

A_mean = (1/n_psu_samples)*sum(x["sample_mean"] for x  in A_calculations.values())
A_total = (data.shape[0]/n_psu_samples)*sum(x["sample_mean"] for x  in A_calculations.values())


########B) Clusters selected  by simplerandom samplign without replacement
selected_clusters_simple = np.random.choice(unique_clusters, size=n_psu_samples, replace=False)
print(selected_clusters_simple)
plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters_simple, title='Soil Data Clustering (Strata)', )
ssu_samples_simple = {}
for cluster in selected_clusters_simple:
    ssu_samples_per_cluster = np.random.choice(clusters[cluster], size=n_ssu_samples, replace=False)
    ssu_samples_simple[cluster] = ssu_samples_per_cluster


ssu_samples_simple_plot = [item for sublist in ssu_samples_simple.values() for item in sublist]
plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters_simple, title='Soil Data Clustering (Strata)', selected_points=ssu_samples_simple_plot )


B_calculations = {}
for cluster, samples in ssu_samples_simple.items():
    sample_cluster_observations = data.loc[samples, 'Cu']
    B_calculations[cluster] = {"total_cluster_size": len(clusters[cluster]), "sample_cluster_size": len(samples), "sample_total": sample_cluster_observations.mean()*len(clusters[cluster]), "sample_mean": sample_cluster_observations.mean(), "sample_variance": sample_cluster_observations.var(ddof=1)}



B_total = (N_clusters/ n_psu_samples) * sum(x["sample_total"] for x  in B_calculations.values())
B_mean = B_total/data.shape[0]


# C) cluster selected by stratified cluster random sampling

n_clusters_per_stratum = 2
n_ssu_samples = 5
selected_clusters_stratum = {}

for stratum in strata:
    selected_clusters_stratified = np.random.choice(a=strata[stratum], size=n_clusters_per_stratum, replace=False)
    selected_clusters_stratum[stratum] = {x:[] for x in selected_clusters_stratified}
    ssu_samples_stratified = []
    for cluster in selected_clusters_stratified:
        ssu_samples_per_cluster = np.random.choice(clusters[cluster], size=n_ssu_samples, replace=False)
        selected_clusters_stratum[stratum][cluster] = ssu_samples_per_cluster

ssu_samples_stratified_plot = [item for sublist in selected_clusters_stratum.values() for subsublist in sublist.values() for item in subsublist]
selected_clusters = [item for sublist in selected_clusters_stratum.values() for item in sublist.keys()]
plot_stratum_grid_highlight(data=data, highlight_clusters=selected_clusters,title='Soil Data Clustering (Strata)', selected_points=ssu_samples_stratified_plot)

C_calculations = {}
for stratum, stratum_clusters in selected_clusters_stratum.items():
    C_calculations[stratum] = {}
    for cluster, samples in stratum_clusters.items():
        sample_cluster_observations = data.loc[samples, 'Cu']
        C_calculations[stratum][cluster] = {
            "total_cluster_size": len(clusters[cluster]),
            "sample_cluster_size": len(samples),
            "sample_total": sum(sample_cluster_observations),
            "sample_mean": sample_cluster_observations.mean(),
            "sample_variance": sample_cluster_observations.var(ddof=1)
        }

for stratum in C_calculations:
    stratum_mean = (1/n_clusters_per_stratum) * sum(x["sample_mean"] for x in C_calculations[stratum].values())
    C_calculations[stratum]["mean"] = stratum_mean

C_mean = (1/N_strata) * sum(x["mean"] for x in C_calculations.values())
C_total = (data.shape[0]) * C_mean


######### D) Simple random sampling of population units, without replacement
D_n_samples = 25
simple_samples = np.random.choice(a = data.index, size=D_n_samples, replace=False)
plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=None, title='Soil Data Clustering (Strata)', selected_points=simple_samples)

D_mean = data.loc[simple_samples, 'Cu'].mean()
D_total = data.loc[simple_samples, 'Cu'].sum()

# Print results
print(f"True total: {True_total}, True mean: {True_mean}")
print(f"A) Total: {A_total}, Mean: {A_mean}")
print(f"B) Total: {B_total}, Mean: {B_mean}")
print(f"C) Total: {C_total}, Mean: {C_mean}")
print(f"D) Total: {D_total}, Mean: {D_mean}")





