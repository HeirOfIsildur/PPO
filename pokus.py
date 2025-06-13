import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from plot_cluster import (
    plot_heatmap,
    plot_cluster_grid,
    plot_cluster_grid_highlight,
    plot_stratum_grid_highlight,
)

#Constants
seed = 45613947 #my UKČO
percentage_white_pixels = 0  # 5% of white pixels
N_clusters = 20
N_strata = 5
random.seed(seed)
np.random.seed(seed)

#Data loading, cleaning, exploration
data = pd.read_csv("soildata.csv", sep=";")
# data = data[["x", "y", "Cu"]]
data = data.replace(',', '', regex=True)
data = data.apply(lambda col: col.astype(int))
plot_heatmap(data, variable_to_plot='Cu', save_path='img/cu_heatmap.pdf')
True_total = data["Cu"].sum()
True_mean = data["Cu"].mean()
# Remove 5% of the rows at random (white pixels, we act like there are not existing at all so number of PSUs doesn't include white pixels)
data = data.sample(frac=1-percentage_white_pixels, random_state=seed).reset_index(drop=True)

#Creation of clusters_to_datapoints
kmeans = KMeans(n_clusters=N_clusters, random_state=seed)
# data['cluster'] = kmeans.fit_predict(data[['x', 'y', 'Al', 'B', 'Ca', 'Fe', 'K', 'Mg', 'Mn/', 'P', 'Zn', 'N','N(min)', 'pH']])
data['cluster'] = kmeans.fit_predict(data[['x', 'y']])
plot_cluster_grid(data = data, plot_variable='cluster', title='Soil Data Clustering', save_path='img/clusters_A.pdf')
clusters_to_datapoints = data.groupby('cluster').apply(lambda df: df.index.tolist()).to_dict()
unique_clusters = list(data['cluster'].unique())

strata_labels = list(range(N_strata))  # [0, 1, 2, 3, 4]

# Map clusters_to_datapoints to strata randomly
clusters_to_stratum = {}
clusters_per_stratum = len(unique_clusters) // N_strata
for i, cluster in enumerate(unique_clusters):
    stratum = strata_labels[i // clusters_per_stratum]
    clusters_to_stratum[cluster] = stratum

data['stratum'] = data['cluster'].map(clusters_to_stratum)
plot_cluster_grid(data=data, plot_variable='stratum', title='Soil Data Strata', save_path='img/simple_strata.pdf')

strata = {}
for cluster, stratum in clusters_to_stratum.items():
    strata.setdefault(stratum, []).append(cluster)

# #Modelling part:
# # A) Clusters selected with probabilities proportional to size, with replacement
events = np.arange(N_clusters)
probabilities = data['cluster'].value_counts(normalize=True).sort_index().values
n_psu_samples = 6
n_ssu_samples = 5
n_clusters_per_stratum = 2
selected_clusters = np.random.choice(events, size=n_psu_samples, replace=False, p=probabilities)
plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters, title='Soil Data Clustering (Strata)', save_path="img/selected_clusters.pdf")

class CalculationsDetails():
    def __init__(self, total_secondary_units: int|None = None, sample_secondary_units: int|None = None, total_primary_units: int|None = None, sample_primary_units: int|None = None, total_zeroth_units: int|None = None, sample_zeroth_units: int|None = None):
        self.total_secondary_units = total_secondary_units
        self.sample_secondary_units = sample_secondary_units
        self.total_primary_units = total_primary_units
        self.sample_primary_units = sample_primary_units
        self.total_zeroth_units = total_zeroth_units
        self.sample_zeroth_units = sample_zeroth_units

        self.children = {}


    def initialize_with_data_points(self, data_points):
        self.children = {data_point: data.loc[data_point, "Cu"] for data_point in data_points}

    def get_all_primary_units(self):
        all_units = []
        for key, value in self.children.items():
            if isinstance(value, CalculationsDetails):
                all_units.extend(value.get_all_primary_units())
            else:
                all_units.append(int(key))
        return all_units

    def get_sample_total(self):
        return float(self.get_sample_mean() * self.total_secondary_units)

    def get_sample_mean(self):
        values = list(self.children.values())
        if not values:
            return None
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            return float(np.mean(values))
        else:
            return float(np.mean([v.get_sample_mean() for v in values]))

    def get_sample_variance(self):
        return (1/self.sample_primary_units) * (1/(self.sample_primary_units-1)) * sum((x.get_sample_mean() - self.get_sample_mean())**2 for x in self.children.values() if isinstance(x, CalculationsDetails))

    def get_sample_total_srswor(self):
        totals_sum = sum(x.get_sample_total() for x in self.children.values() if isinstance(x, CalculationsDetails))
        return totals_sum/self.sample_primary_units * self.total_primary_units

    def get_sample_mean_srswor(self):
        return self.get_sample_total_srswor()/ self.total_secondary_units

    def get_sample_total_variance_srswor(self):
        left_summand = self.total_primary_units * (self.total_primary_units - self.sample_primary_units) * self.get_su2() / self.sample_primary_units
        right_summand = self.total_primary_units/self.sample_primary_units*sum(x.get_partial_si2_inside_sum() for x in self.children.values())
        return left_summand + right_summand

    def get_sample_variance_srswor(self):
        return (1/(self.total_secondary_units**2)) * self.get_sample_total_variance_srswor()

    def get_su2(self):
        cluster_sample_totals = [x.get_sample_total() for x in self.children.values()]
        return self.calculate_simples_math_sample_variance(cluster_sample_totals)

    def get_partial_si2_inside_sum(self):
        return self.total_secondary_units * (self.total_secondary_units - self.sample_secondary_units) * self.calculate_simples_math_sample_variance(values = list(self.children.values())) / self.sample_secondary_units

    def calculate_simples_math_sample_variance(self, values):
        mean = float(np.mean(values))
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        n = len(values)
        return (1/(n - 1)) * squared_diff_sum

    def get_sample_total_stratified(self):
        return sum(x.get_sample_total_srswor() for x in self.children.values())

    def get_sample_mean_stratified(self):
        return self.get_sample_total_stratified()/ self.total_secondary_units

    def get_sample_total_variance_stratified(self):
        return sum(x.get_sample_total_variance_srswor() for x in self.children.values())

    def get_sample_mean_variance_stratified(self):
        return self.get_sample_total_variance_stratified() / (self.total_secondary_units**2)

A = CalculationsDetails(total_secondary_units=len(data), sample_primary_units=n_psu_samples, total_primary_units=N_clusters)

for i, cluster in enumerate(selected_clusters):
    cluster_underscore = f"{cluster}_{i}"
    A.children[cluster_underscore] = CalculationsDetails(total_secondary_units = len(clusters_to_datapoints[cluster]), sample_secondary_units = n_ssu_samples)
    ssu_samples_per_cluster = np.random.choice(clusters_to_datapoints[cluster], size=n_ssu_samples, replace=False)
    A.children[cluster_underscore].initialize_with_data_points(ssu_samples_per_cluster)

plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters, title='Soil Data Clustering (Strata)', selected_points=A.get_all_primary_units(), save_path='img/selected_points_A.pdf')
print(A.get_sample_mean(), A.get_sample_variance())

#######B) Clusters selected  by simplerandom samplign without replacement
selected_clusters_simple = np.random.choice(unique_clusters, size=n_psu_samples, replace=False)
plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters_simple, title='Soil Data Clustering (Strata)', )


B = CalculationsDetails(total_secondary_units = len(data), sample_primary_units = n_psu_samples, total_primary_units = N_clusters)
for cluster in selected_clusters_simple:
    B.children[cluster] = CalculationsDetails(total_secondary_units = len(clusters_to_datapoints[cluster]), sample_secondary_units = n_ssu_samples)
    ssu_samples_per_cluster = np.random.choice(
        clusters_to_datapoints[cluster], size=n_ssu_samples, replace=False
    )
    B.children[cluster].initialize_with_data_points(ssu_samples_per_cluster)


print(B.get_sample_mean_srswor(), B.get_sample_variance_srswor())

plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters_simple, title='Soil Data Clustering (Strata)', selected_points=B.get_all_primary_units() )

# C) cluster selected by stratified cluster random sampling


selected_clusters_stratum = {}

for stratum in strata:
    selected_clusters_stratified = np.random.choice(a=strata[stratum], size=n_clusters_per_stratum, replace=False)
    selected_clusters_stratum[stratum] = {x:[] for x in selected_clusters_stratified}
    ssu_samples_stratified = []
    for cluster in selected_clusters_stratified:
        ssu_samples_per_cluster = np.random.choice(clusters_to_datapoints[cluster], size=n_ssu_samples, replace=False)
        selected_clusters_stratum[stratum][cluster] = ssu_samples_per_cluster

ssu_samples_stratified_plot = [item for sublist in selected_clusters_stratum.values() for subsublist in sublist.values() for item in subsublist]
selected_clusters = [item for sublist in selected_clusters_stratum.values() for item in sublist.keys()]
plot_stratum_grid_highlight(data=data, highlight_clusters=selected_clusters,title='Soil Data Clustering (Strata)', selected_points=ssu_samples_stratified_plot)

C= CalculationsDetails(total_secondary_units = len(data), total_zeroth_units = N_strata, sample_zeroth_units = n_clusters_per_stratum)
for stratum in strata:
    C.children[stratum] = CalculationsDetails(total_primary_units = len(strata[stratum]), sample_primary_units = n_clusters_per_stratum, total_secondary_units = sum(len(clusters_to_datapoints[cluster]) for cluster in strata[stratum]), sample_secondary_units = n_ssu_samples * n_clusters_per_stratum)
    for cluster, samples in selected_clusters_stratum[stratum].items():
        C.children[stratum].children[cluster] = CalculationsDetails(total_secondary_units = len(clusters_to_datapoints[cluster]), sample_secondary_units = len(samples))
        C.children[stratum].children[cluster].initialize_with_data_points(samples)

print(C.get_sample_mean_stratified(), C.get_sample_mean_variance_stratified())


# C_calculations = {}
# for stratum, stratum_clusters in selected_clusters_stratum.items():
#     C_calculations[stratum] = {}
#     C_calculations[stratum]["total_stratum_size"] = sum(len(clusters_to_datapoints[cluster]) for cluster in strata[stratum])
#     for cluster, samples in stratum_clusters.items():
#         sample_cluster_observations = data.loc[samples, 'Cu']
#         print(sample_cluster_observations)
#         C_calculations[stratum][cluster] = {
#             "total_cluster_size": len(clusters_to_datapoints[cluster]),
#             "sample_cluster_size": len(samples),
#             "sample_total": sum(sample_cluster_observations),
#             "sample_mean": sample_cluster_observations.mean(),
#             "sample_variance": sample_cluster_observations.var(ddof=1)
#         }
#     C_calculations[stratum]["sample_mean"] = (1/n_clusters_per_stratum) * sum(x["sample_mean"] for x in C_calculations[stratum].values() if isinstance(x, dict))
#
# for stratum in strata:
#     N_clusters_strata = int(N_clusters/ N_strata)
#     C_calculations_stratum_clusters = {
#         k: v
#         for k, v in C_calculations[stratum].items()
#         if isinstance(k, (int, float, np.integer, np.floating))
#     }
#     C_calculations[stratum]["sample_variance"] = var_B(N_clusters_strata, n_clusters_per_stratum, C_calculations_stratum_clusters, C_calculations[stratum]["sample_mean"], selected_clusters_stratum[stratum])
#
#
# sum = 0
# for stratum in C_calculations:
#     strata_variance = C_calculations[stratum]["sample_variance"]
#     total_stratum_size = C_calculations[stratum]["total_stratum_size"]
#     sample_stratum_size = n_clusters_per_stratum * n_ssu_samples
#     sum += total_stratum_size * (total_stratum_size - sample_stratum_size) * strata_variance / sample_stratum_size
#
#
# C_mean = (1/N_strata) * np.sum([x["sample_mean"] for x in C_calculations.values()])
# C_total = (data.shape[0]) * C_mean
# C_total_variance = sum
# C_mean_variance = C_total_variance / (data.shape[0]**2)
# C_mean_standard_error = np.sqrt(C_mean_variance)




# sum(x["total_stratum_size"]*(x["total_stratum_size"]-0)*x["sample_variance"] for stratum in C_calculations for x in C_calculations[stratum].values() if isinstance(x, dict))

######### D) Simple random sampling of population units, without replacement
D_n_samples = n_psu_samples*n_ssu_samples
D = CalculationsDetails(total_secondary_units = len(data), sample_secondary_units = D_n_samples)
simple_samples = np.random.choice(a = data.index, size=D_n_samples, replace=False)
plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=None, title='Soil Data Clustering (Strata)', selected_points=simple_samples)

D.initialize_with_data_points(simple_samples)

print(D.get_sample_mean(), (1-D.sample_secondary_units/D.total_secondary_units)*(D.calculate_simples_math_sample_variance(values=data.loc[simple_samples, 'Cu'])) / D.sample_secondary_units)