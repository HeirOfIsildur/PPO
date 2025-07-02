import itertools
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
all_percentages_white_points = (0.0)
possible_combinations = list(itertools.product(all_n_psu_per_stratum, all_n_ssu_samples))

N_clusters = 20
N_strata = 5
percentage_white_pixels = 0.0
n_repetition = 100

full_df = pd.DataFrame()


# Data loading, cleaning, exploration
data_original = pd.read_csv("soildata.csv", sep=";")
# data_original = data_original[["x", "y", "Cu"]]
data_original = data_original.replace(",", "", regex=True)
data_original = data_original.apply(lambda col: col.astype(int))
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

strata = {}
for cluster, stratum in clusters_to_stratum.items():
    strata.setdefault(stratum, []).append(cluster)

all_results = []
for combination in possible_combinations:
    n_psu_samples_per_stratum, n_ssu_samples = combination
    n_psu_samples = n_psu_samples_per_stratum * N_strata

    data = data_original.sample(frac=1-percentage_white_pixels, random_state=seed).reset_index(drop=True).copy()
    for iteration in range(n_repetition):
        events = np.arange(N_clusters)
        probabilities = data['cluster'].value_counts(normalize=True).sort_index().values

        selected_clusters = np.random.choice(events, size=n_psu_samples, replace=False, p=probabilities)

        A = CalculationsDetails(total_secondary_units=len(data), sample_primary_units=n_psu_samples, total_primary_units=N_clusters)

        for i, cluster in enumerate(selected_clusters):
            cluster_underscore = f"{cluster}_{i}"
            clusters_to_datapoints = (
                data.groupby("cluster").apply(lambda df: df.index.tolist()).to_dict()
            )
            A.children[cluster_underscore] = CalculationsDetails(total_secondary_units = len(clusters_to_datapoints[cluster]), sample_secondary_units = n_ssu_samples)
            ssu_samples_per_cluster = np.random.choice(clusters_to_datapoints[cluster], size=n_ssu_samples, replace=False)
            A.children[cluster_underscore].initialize_with_data_points(data= data, data_points=ssu_samples_per_cluster)


        #######B) Clusters selected  by simplerandom samplign without replacement
        selected_clusters_simple = np.random.choice(unique_clusters, size=n_psu_samples, replace=False)
        # plot_cluster_grid_highlight(
        #     data=data,
        #     plot_variable="cluster",
        #     highlight_clusters=selected_clusters,
        #     title="Soil Data Clustering (Strata)",
        #     selected_points=A.get_all_primary_units(),
        #     save_path="img/selected_points_A.pdf",
        # )

        B = CalculationsDetails(total_secondary_units = len(data), sample_primary_units = n_psu_samples, total_primary_units = N_clusters)
        for cluster in selected_clusters_simple:
            B.children[cluster] = CalculationsDetails(total_secondary_units = len(clusters_to_datapoints[cluster]), sample_secondary_units = n_ssu_samples)
            ssu_samples_per_cluster = np.random.choice(
                clusters_to_datapoints[cluster], size=n_ssu_samples, replace=False
            )
            B.children[cluster].initialize_with_data_points(data = data, data_points = ssu_samples_per_cluster)


        print(round(B.get_sample_mean_srswor(), 2), round(B.get_sample_variance_srswor(), 2))


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

        D.initialize_with_data_points(data=data, data_points=simple_samples)

        print(round(D.get_sample_mean(),2), round((1-D.sample_secondary_units/D.total_secondary_units)*(D.calculate_simples_math_sample_variance(values=data.loc[simple_samples, 'Cu'])) / D.sample_secondary_units, 2))

        results = {
            "n_psu_samples": n_psu_samples,
            "n_ssu_samples": n_ssu_samples,
            "n_psu_samples_per_stratum": n_psu_samples_per_stratum,
            "percentage_white_pixels": percentage_white_pixels,
            "iteration": iteration,
            "A_mean": round(A.get_sample_mean(), 2),
            "B_mean": round(B.get_sample_mean_srswor(), 2),
            "C_mean": round(C.get_sample_mean_stratified(), 2),
            "D_mean": round(D.get_sample_mean(), 2),
        }
        all_results.append(results)

summary_df = pd.DataFrame(all_results)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(summary_df)

summary_df.groupby(['n_psu_samples', 'n_ssu_samples']).mean().reset_index()
for combination in possible_combinations:
    n_psu_samples_per_stratum, n_ssu_samples = combination
    n_psu_samples = n_psu_samples_per_stratum * N_strata
    print(f"{n_psu_samples =}, {n_ssu_samples=}")
    summary_df_subset = summary_df[(summary_df['n_psu_samples'] == n_psu_samples) & (summary_df['n_ssu_samples'] == n_ssu_samples)].copy()
    mean_error_A = summary_df_subset['A_mean'].mean() - True_mean
    mean_absolute_error_A = mean_absolute_error(summary_df_subset['A_mean'], [True_mean] * len(summary_df_subset))
    variance_A = summary_df_subset['A_mean'].var(ddof=1)

    mean_error_B = summary_df_subset['B_mean'].mean() - True_mean
    mean_absolute_error_B = mean_absolute_error(summary_df_subset['B_mean'], [True_mean] * len(summary_df_subset))
    variance_B = summary_df_subset['B_mean'].var(ddof=1)

    mean_error_C = summary_df_subset['C_mean'].mean() - True_mean
    mean_absolute_error_C = mean_absolute_error(summary_df_subset['C_mean'], [True_mean] * len(summary_df_subset))
    variance_C = summary_df_subset['C_mean'].var(ddof=1)

    mean_error_D = summary_df_subset['D_mean'].mean() - True_mean
    mean_absolute_error_D = mean_absolute_error(summary_df_subset['D_mean'], [True_mean] * len(summary_df_subset))
    variance_D = summary_df_subset['D_mean'].var(ddof=1)


    print(f"{'Bias A:':<12} {round(mean_error_A, 1):<12} {'MAE:':<6} {round(mean_absolute_error_A, 1):<12} {'Variance A:':<12} {round(variance_A, 1):<12}")
    print(f"{'Bias B:':<12} {round(mean_error_B, 1):<12} {'MAE:':<6} {round(mean_absolute_error_B, 1):<12}  {'Variance B:':<12} {round(variance_B, 1):<12}")
    print(f"{'Bias C:':<12} {round(mean_error_C, 1):<12} {'MAE:':<6} {round(mean_absolute_error_C, 1):<12}  {'Variance C:':<12} {round(variance_C, 1):<12}")
    print(f"{'Bias D:':<12} {round(mean_error_D, 1):<12} {'MAE:':<6} {round(mean_absolute_error_D, 1):<12} {'Variance D:':<12} {round(variance_D, 1):<12}")