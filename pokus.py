import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from plot_cluster import (
    plot_cluster_grid,
    plot_cluster_grid_highlight,
    plot_stratum_grid_highlight,
)

#Constants
seed = 1998 #my UKČO
percentage_white_pixels = 0
N_clusters = 20
N_strata = 5
random.seed(seed)
np.random.seed(seed)

#Data loading, cleaning, exploration
data = pd.read_csv("soildata.csv", sep=";")
data = data[["x", "y", "Cu"]]
data["Cu"] = data["Cu"].str.replace(",", "", regex=False).astype(int)
True_total = data["Cu"].sum()
True_mean = data["Cu"].mean()
# Remove 5% of the rows at random (white pixels, we act like there are not existing at all so number of PSUs doesn't include white pixels)
data = data.sample(frac=1-percentage_white_pixels, random_state=seed).reset_index(drop=True)

#Creation of clusters_to_datapoints
kmeans = KMeans(n_clusters=N_clusters, random_state=seed)
data['cluster'] = kmeans.fit_predict(data[['x', 'y']])
plot_cluster_grid(data = data, plot_variable='cluster', title='Soil Data Clustering')
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
plot_cluster_grid(data=data, plot_variable='stratum', title='Soil Data Clustering (Strata)')

strata = {}
for cluster, stratum in clusters_to_stratum.items():
    strata.setdefault(stratum, []).append(cluster)

# #Modelling part:
# # A) Clusters selected with probabilities proportional to size, with replacement
events = np.arange(N_clusters)
probabilities = data['cluster'].value_counts(normalize=True).sort_index().values
n_psu_samples = 2
n_ssu_samples = 2
selected_clusters = np.random.choice(events, size=n_psu_samples, replace=True, p=probabilities)
plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters, title='Soil Data Clustering (Strata)', )

class CalculationsDetails():
    def __init__(self):
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






# A = CalculationsDetails()
# A.total_secondary_units = len(data)
# A.sample_primary_units = n_psu_samples
# A.total_primary_units = N_clusters
#
# for cluster in selected_clusters:
#     A.children[cluster] = CalculationsDetails()
#     A.children[cluster].total_secondary_units = len(clusters_to_datapoints[cluster])
#     A.children[cluster].sample_secondary_units = n_psu_samples
#     ssu_samples_per_cluster = np.random.choice(clusters_to_datapoints[cluster], size=n_ssu_samples, replace=False)
#     A.children[cluster].initialize_with_data_points(ssu_samples_per_cluster)
#
# plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters, title='Soil Data Clustering (Strata)', selected_points=A.get_all_primary_units())
# print(A.get_sample_mean(), A.get_sample_variance())
#
# ((A.children[11].get_sample_total()+A.children[15].get_sample_total())/2*20)/ data.shape[0]
# A.get_sample_mean_srswor()
# A.children[11].get_partial_si2_inside_sum()
# A.get_sample_variance_srswor()
# print(A.get_sample_mean_srswor(), A.get_sample_variance_srswor())


########B) Clusters selected  by simplerandom samplign without replacement
selected_clusters_simple = np.random.choice(unique_clusters, size=n_psu_samples, replace=False)
plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters_simple, title='Soil Data Clustering (Strata)', )


B = CalculationsDetails()
B.total_secondary_units = len(data)
B.sample_primary_units = n_psu_samples
B.total_primary_units = N_clusters
for cluster in selected_clusters:
    B.children[cluster] = CalculationsDetails()
    B.children[cluster].total_secondary_units = len(clusters_to_datapoints[cluster])
    B.children[cluster].sample_secondary_units = n_ssu_samples
    ssu_samples_per_cluster = np.random.choice(
        clusters_to_datapoints[cluster], size=n_ssu_samples, replace=False
    )
    B.children[cluster].initialize_with_data_points(ssu_samples_per_cluster)


print(B.get_sample_mean_srswor(), B.get_sample_variance_srswor())

plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters_simple, title='Soil Data Clustering (Strata)', selected_points=B.get_all_primary_units() )


B_calculations = {}
for cluster, samples in ssu_samples_simple.items():
    sample_cluster_observations = data.loc[samples, 'Cu']
    B_calculations[cluster] = {"total_cluster_size": len(clusters_to_datapoints[cluster]), "sample_cluster_size": len(samples), "sample_total": sample_cluster_observations.mean() * len(clusters_to_datapoints[cluster]), "sample_mean": sample_cluster_observations.mean(), "sample_variance": sample_cluster_observations.var(ddof=1)}




B_total = (N_clusters/ n_psu_samples) * sum(x["sample_total"] for x  in B_calculations.values())
B_mean = B_total/data.shape[0]



def var_B(N_clusters, n_psu_samples, B_calculations, B_mean, ssu_samples_simple):
    s_u2 = (
        1
        / (n_psu_samples - 1)
        * sum((x["sample_total"] - B_mean) ** 2 for x in B_calculations.values())
        #TODO: Opravdu tady ma byt sample total??
    )

    first_part = N_clusters*(N_clusters - n_psu_samples) * s_u2/ n_psu_samples
    second_part = N_clusters/n_psu_samples *sum(y_func(B_calculations[cluster], sample_cluster_observations) for (cluster, samples) in ssu_samples_simple.items())
    return first_part + second_part

def y_func(B_calculation, sample_cluster_observations):
    Mi = B_calculation["total_cluster_size"]
    mi = B_calculation["sample_cluster_size"]
    s_i2 = s_i2_calculation(B_calculation, sample_cluster_observations)
    return Mi*(Mi-mi) * s_i2 / mi

def s_i2_calculation(B_calculation, sample_cluster_observations):
    mi = B_calculation["total_cluster_size"]
    yi_hat = B_calculation["sample_mean"]
    return (1/(mi-1)) * sum([(x-yi_hat)**2 for x in sample_cluster_observations])

B_total_variance_book = var_B(N_clusters, n_psu_samples, B_calculations, B_mean, ssu_samples_simple)
B_mean_variance_book = B_total_variance_book/ (data.shape[0]**2)
B_mean_standard_error_book = np.sqrt(B_mean_variance_book)

# C) cluster selected by stratified cluster random sampling

n_clusters_per_stratum = 2
n_ssu_samples = 2
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

C_calculations = {}
for stratum, stratum_clusters in selected_clusters_stratum.items():
    C_calculations[stratum] = {}
    C_calculations[stratum]["total_stratum_size"] = sum(len(clusters_to_datapoints[cluster]) for cluster in strata[stratum])
    for cluster, samples in stratum_clusters.items():
        sample_cluster_observations = data.loc[samples, 'Cu']
        print(sample_cluster_observations)
        C_calculations[stratum][cluster] = {
            "total_cluster_size": len(clusters_to_datapoints[cluster]),
            "sample_cluster_size": len(samples),
            "sample_total": sum(sample_cluster_observations),
            "sample_mean": sample_cluster_observations.mean(),
            "sample_variance": sample_cluster_observations.var(ddof=1)
        }
    C_calculations[stratum]["sample_mean"] = (1/n_clusters_per_stratum) * sum(x["sample_mean"] for x in C_calculations[stratum].values() if isinstance(x, dict))

for stratum in strata:
    N_clusters_strata = int(N_clusters/ N_strata)
    C_calculations_stratum_clusters = {
        k: v
        for k, v in C_calculations[stratum].items()
        if isinstance(k, (int, float, np.integer, np.floating))
    }
    C_calculations[stratum]["sample_variance"] = var_B(N_clusters_strata, n_clusters_per_stratum, C_calculations_stratum_clusters, C_calculations[stratum]["sample_mean"], selected_clusters_stratum[stratum])


sum = 0
for stratum in C_calculations:
    strata_variance = C_calculations[stratum]["sample_variance"]
    total_stratum_size = C_calculations[stratum]["total_stratum_size"]
    sample_stratum_size = n_clusters_per_stratum * n_ssu_samples
    sum += total_stratum_size * (total_stratum_size - sample_stratum_size) * strata_variance / sample_stratum_size


C_mean = (1/N_strata) * np.sum([x["sample_mean"] for x in C_calculations.values()])
C_total = (data.shape[0]) * C_mean
C_total_variance = sum
C_mean_variance = C_total_variance / (data.shape[0]**2)
C_mean_standard_error = np.sqrt(C_mean_variance)




# sum(x["total_stratum_size"]*(x["total_stratum_size"]-0)*x["sample_variance"] for stratum in C_calculations for x in C_calculations[stratum].values() if isinstance(x, dict))


######### D) Simple random sampling of population units, without replacement
D_n_samples = 4
simple_samples = np.random.choice(a = data.index, size=D_n_samples, replace=False)
plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=None, title='Soil Data Clustering (Strata)', selected_points=simple_samples)

D_mean = data.loc[simple_samples, 'Cu'].mean()
D_total = data.loc[simple_samples, 'Cu'].sum()
D_mean_variance = (1-(D_n_samples/data.shape[0])) * ((1/(D_n_samples-1)) * np.sum((data.loc[simple_samples, 'Cu'] - D_mean)**2)) / D_n_samples
D_total_variance = (data.shape[0]**2) * (1 - (D_n_samples/data.shape[0])) * D_mean_variance / D_n_samples
D_mean_standard_error = np.sqrt(D_mean_variance)


# Print results
print(f"True total: {True_total}, True mean: {True_mean}")
print(f"A) Total: {A_total}, Mean: {A_mean}", f"Variance: {A_mean_variance_book}", f"Variance book: {A_mean_variance_book}")
print(f"B) Total: {B_total}, Mean: {B_mean}", f"Variance: {B_mean_variance_book}", f"Standard Error: {B_mean_standard_error_book}")
print(f"C) Total: {C_total}, Mean: {C_mean}", f"Variance: {C_mean_variance}", f"Standard Error: {C_mean_standard_error}")
print(f"D) Total: {D_total}, Mean: {D_mean}", f"Variance: {D_mean_variance}", f"Standard Error: {D_mean_standard_error}")





