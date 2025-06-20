import itertools
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from calculation_details import CalculationsDetails
from plot_cluster import (
    plot_heatmap,
    plot_cluster_grid,
    plot_cluster_grid_highlight,
    plot_stratum_grid_highlight,
)

#Constants
seed = 45613947 #my UKČO
random.seed(seed)
np.random.seed(seed)

all_n_psu_per_stratum = (2, 3)
all_n_ssu_samples = (2, 5, 10)
all_percentages_white_points = (0.0, 0.1, 0.5)
possible_combinations = list(itertools.product(all_n_psu_per_stratum, all_n_ssu_samples, all_percentages_white_points))

N_clusters = 20
N_strata = 5


# Data loading, cleaning, exploration
data_original = pd.read_csv("soildata.csv", sep=";")
# data_original = data_original[["x", "y", "Cu"]]
data_original = data_original.replace(",", "", regex=True)
data_original = data_original.apply(lambda col: col.astype(int))
plot_heatmap(data_original, variable_to_plot="Cu", save_path="img/cu_heatmap.pdf")
True_total = data_original["Cu"].sum()
True_mean = data_original["Cu"].mean()
data_original["Cu"].describe()

data_original = data_original.sample(frac=1, random_state=seed).reset_index(
    drop=True
)

# Creation of clusters_to_datapoints
kmeans = KMeans(n_clusters=N_clusters, random_state=seed)
# data_original['cluster'] = kmeans.fit_predict(data_original[['x', 'y', 'Al', 'B', 'Ca', 'Fe', 'K', 'Mg', 'Mn/', 'P', 'Zn', 'N','N(min)', 'pH']])
data_original["cluster"] = kmeans.fit_predict(data_original[["x", "y"]])
plot_cluster_grid(
    data=data_original,
    plot_variable="cluster",
    title="Soil Data Clustering",
    save_path="img/clusters_A.pdf",
)

unique_clusters = list(data_original["cluster"].unique())

strata_labels = list(range(N_strata))  # [0, 1, 2, 3, 4]
stratum_to_clusters = {
    0: [4, 11, 14, 18],
    1: [2, 7, 8, 15],
    2: [0, 10, 13, 19],
    3: [3, 5, 12, 16],
    4: [1, 6, 9, 17],
}
clusters_to_stratum = {
    cluster: stratum
    for stratum, clusters in stratum_to_clusters.items()
    for cluster in clusters
}

data_original["stratum"] = data_original["cluster"].map(clusters_to_stratum)
plot_cluster_grid(
    data=data_original,
    plot_variable="stratum",
    title="Soil Data Strata",
    save_path="img/simple_strata.pdf",
)

strata = {}
for cluster, stratum in clusters_to_stratum.items():
    strata.setdefault(stratum, []).append(cluster)

all_results = []
for combination in possible_combinations:
    n_psu_samples_per_stratum, n_ssu_samples, percentage_white_pixels = combination
    n_psu_samples = n_psu_samples_per_stratum * N_strata

    data = data_original.sample(frac=1-percentage_white_pixels, random_state=seed).reset_index(drop=True).copy()

    # #Modelling part:
    # # A) Clusters selected with probabilities proportional to size, with replacement
    events = np.arange(N_clusters)
    probabilities = data['cluster'].value_counts(normalize=True).sort_index().values

    selected_clusters = np.random.choice(events, size=n_psu_samples, replace=False, p=probabilities)
    plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters, title='Soil Data Clustering (Strata)', save_path="img/selected_clusters_A.pdf")

    A = CalculationsDetails(total_secondary_units=len(data), sample_primary_units=n_psu_samples, total_primary_units=N_clusters)

    for i, cluster in enumerate(selected_clusters):
        cluster_underscore = f"{cluster}_{i}"
        clusters_to_datapoints = (
            data.groupby("cluster").apply(lambda df: df.index.tolist()).to_dict()
        )
        A.children[cluster_underscore] = CalculationsDetails(total_secondary_units = len(clusters_to_datapoints[cluster]), sample_secondary_units = n_ssu_samples)
        ssu_samples_per_cluster = np.random.choice(clusters_to_datapoints[cluster], size=n_ssu_samples, replace=False)
        A.children[cluster_underscore].initialize_with_data_points(data= data, data_points=ssu_samples_per_cluster)

    plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters, title='Soil Data Clustering (Strata)', selected_points=A.get_all_primary_units(), save_path='img/selected_points_A.pdf')
    print(round(A.get_sample_mean(),2), round(A.get_sample_variance(),2))

    #######B) Clusters selected  by simplerandom samplign without replacement
    selected_clusters_simple = np.random.choice(unique_clusters, size=n_psu_samples, replace=False)
    plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters_simple, title='Soil Data Clustering (Strata)', save_path="img/selected_clusters_B.pdf")


    B = CalculationsDetails(total_secondary_units = len(data), sample_primary_units = n_psu_samples, total_primary_units = N_clusters)
    for cluster in selected_clusters_simple:
        B.children[cluster] = CalculationsDetails(total_secondary_units = len(clusters_to_datapoints[cluster]), sample_secondary_units = n_ssu_samples)
        ssu_samples_per_cluster = np.random.choice(
            clusters_to_datapoints[cluster], size=n_ssu_samples, replace=False
        )
        B.children[cluster].initialize_with_data_points(data = data, data_points = ssu_samples_per_cluster)


    print(round(B.get_sample_mean_srswor(), 2), round(B.get_sample_variance_srswor(), 2))

    plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=selected_clusters_simple, title='Soil Data Clustering (Strata)', selected_points=B.get_all_primary_units(), save_path='img/selected_points_B.pdf')

    # C) cluster selected by stratified cluster random sampling


    selected_clusters_stratum = {}

    for stratum in strata:
        selected_clusters_stratified = np.random.choice(a=strata[stratum], size=n_psu_samples_per_stratum, replace=False)
        selected_clusters_stratum[stratum] = {x:[] for x in selected_clusters_stratified}
        ssu_samples_stratified = []
        for cluster in selected_clusters_stratified:
            ssu_samples_per_cluster = np.random.choice(clusters_to_datapoints[cluster], size=n_ssu_samples, replace=False)
            selected_clusters_stratum[stratum][cluster] = ssu_samples_per_cluster


    ssu_samples_stratified_plot = [item for sublist in selected_clusters_stratum.values() for subsublist in sublist.values() for item in subsublist]
    selected_clusters = [item for sublist in selected_clusters_stratum.values() for item in sublist.keys()]
    plot_stratum_grid_highlight(data=data, highlight_clusters=selected_clusters,title='Soil Data Clustering (Strata)', selected_points=ssu_samples_stratified_plot, save_path='img/selected_points_C.pdf')

    C= CalculationsDetails(total_secondary_units = len(data), total_zeroth_units = N_strata, sample_zeroth_units = n_psu_samples_per_stratum)
    for stratum in strata:
        C.children[stratum] = CalculationsDetails(total_primary_units = len(strata[stratum]), sample_primary_units = n_psu_samples_per_stratum, total_secondary_units = sum(len(clusters_to_datapoints[cluster]) for cluster in strata[stratum]), sample_secondary_units = n_ssu_samples * n_psu_samples_per_stratum)
        for cluster, samples in selected_clusters_stratum[stratum].items():
            C.children[stratum].children[cluster] = CalculationsDetails(total_secondary_units = len(clusters_to_datapoints[cluster]), sample_secondary_units = len(samples))
            C.children[stratum].children[cluster].initialize_with_data_points(data=data, data_points=samples)

    print(round(C.get_sample_mean_stratified(),2), round(C.get_sample_mean_variance_stratified(),2))

    ######### D) Simple random sampling of population units, without replacement
    D_n_samples = n_psu_samples*n_ssu_samples
    D = CalculationsDetails(total_secondary_units = len(data), sample_secondary_units = D_n_samples)
    simple_samples = np.random.choice(a = data.index, size=D_n_samples, replace=False)
    plot_cluster_grid_highlight(data=data, plot_variable='cluster', highlight_clusters=None, title='Soil Data Clustering (Strata)', selected_points=simple_samples, save_path='img/selected_points_D.pdf')

    D.initialize_with_data_points(data=data, data_points=simple_samples)

    print(round(D.get_sample_mean(),2), round((1-D.sample_secondary_units/D.total_secondary_units)*(D.calculate_simples_math_sample_variance(values=data.loc[simple_samples, 'Cu'])) / D.sample_secondary_units, 2))


    results = {
        "n_psu_samples": n_psu_samples,
        "n_ssu_samples": n_ssu_samples,
        "n_psu_samples_per_stratum": n_psu_samples_per_stratum,
        "percentage_white_pixels": percentage_white_pixels,
        "A_mean": round(A.get_sample_mean(), 2),
        "A_variance": round(A.get_sample_variance(), 2),
        "B_mean": round(B.get_sample_mean_srswor(), 2),
        "B_variance": round(B.get_sample_variance_srswor(), 2),
        "C_mean": round(C.get_sample_mean_stratified(), 2),
        "C_variance": round(C.get_sample_mean_variance_stratified(), 2),
        "D_mean": round(D.get_sample_mean(), 2),
        "D_variance": round(
            (1 - D.sample_secondary_units / D.total_secondary_units)
            * D.calculate_simples_math_sample_variance(
                values=data.loc[simple_samples, "Cu"]
            )
            / D.sample_secondary_units,
            2,
        ),
    }
    all_results.append(results)

summary_df = pd.DataFrame(all_results)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(summary_df)

