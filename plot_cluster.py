import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

def plot_cluster_grid(data, plot_variable, title):
    pivot = data.pivot(index='y', columns='x', values=plot_variable)
    img = pivot.values

    # Get unique codes and their corresponding labels if plotting strata
    if plot_variable == 'stratum_code' and 'stratum' in data.columns:
        code_to_label = dict(enumerate(pd.Categorical(data['stratum']).categories))
        unique_codes = np.unique(img[~np.isnan(img)]).astype(int)
        legend_labels = [code_to_label[code] for code in unique_codes]
    else:
        unique_codes = np.unique(img[~np.isnan(img)]).astype(int)
        legend_labels = [str(code) for code in unique_codes]

    cmap = plt.cm.get_cmap('tab20', len(unique_codes))

    plt.figure(figsize=(10, 5))
    plt.imshow(img, cmap=cmap, origin='lower', interpolation='none', aspect='auto')

    ax = plt.gca()
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    xtick_labels = [str(val) if i % 3 == 0 else '' for i, val in enumerate(pivot.columns)]
    ytick_labels = [str(val) if i % 3 == 0 else '' for i, val in enumerate(pivot.index)]
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(ytick_labels)

    ax.set_xticks(np.arange(-0.5, img.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, img.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)

    handles = [
        Patch(facecolor=cmap(i), edgecolor='none', label=legend_labels[i])
        for i in range(len(unique_codes))
    ]
    ncol = max(1, (len(unique_codes) + 9) // 10)
    plt.legend(
        handles=handles,
        title='Stratum' if plot_variable == 'stratum' else 'Cluster',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        ncol=ncol
    )
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    ax.set_title(title, pad=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()


def plot_cluster_grid_highlight(
    data, plot_variable, highlight_clusters, title, selected_points=None
):
    import matplotlib.colors as mcolors

    # Prepare selection mask
    if selected_points is not None and len(selected_points) > 0:
        if isinstance(selected_points, (list, np.ndarray)):
            sel_idx = set(data.iloc[selected_points].index)
        else:
            sel_idx = set(selected_points.index)
    else:
        sel_idx = set()

    pivot = data.pivot(index='y', columns='x', values=plot_variable)
    img = pivot.values

    unique_codes = np.unique(img[~np.isnan(img)]).astype(int)
    legend_labels = [str(code) for code in unique_codes]

    # Only highlight clusters_to_datapoints if highlight_clusters is not None
    if highlight_clusters is not None:
        highlight_set = set(highlight_clusters)
    else:
        highlight_set = set()
    code_to_idx = {code: i for i, code in enumerate(unique_codes)}
    nan_idx = len(unique_codes)

    base_cmap = plt.cm.get_cmap('tab20', len(unique_codes))

    # Build an RGBA image for the grid
    rgba_img = np.ones((*img.shape, 4))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = img[i, j]
            if np.isnan(val):
                rgba_img[i, j] = (1, 1, 1, 0)
                continue
            code = int(val)
            color = list(base_cmap(code_to_idx[code]))
            # Find the original DataFrame index for this cell
            x = pivot.columns[j]
            y = pivot.index[i]
            idx = data[(data['x'] == x) & (data['y'] == y)].index
            if len(idx) > 0 and idx[0] in sel_idx:
                color[-1] = 1.0
            elif highlight_set and code in highlight_set:
                color[-1] = 0.5
            else:
                color[-1] = 0.1
            rgba_img[i, j] = color

    plt.figure(figsize=(10, 5))
    plt.imshow(rgba_img, origin='lower', interpolation='none', aspect='auto')
    ax = plt.gca()
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    xtick_labels = [str(val) if i % 3 == 0 else '' for i, val in enumerate(pivot.columns)]
    ytick_labels = [str(val) if i % 3 == 0 else '' for i, val in enumerate(pivot.index)]
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(ytick_labels)
    ax.set_xticks(np.arange(-0.5, img.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, img.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)

    # Legend
    handles = [
        Patch(facecolor=base_cmap(i), edgecolor='none', label=legend_labels[i])
        for i in range(len(unique_codes))
    ]
    ncol = max(1, (len(unique_codes) + 9) // 10)
    plt.legend(
        handles=handles,
        title='Cluster',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        ncol=ncol
    )
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    ax.set_title(title, pad=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()


def plot_stratum_grid_highlight(
    data, highlight_clusters, title, selected_points=None
):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    plot_variable = 'stratum'
    pivot = data.pivot(index='y', columns='x', values=plot_variable)
    img = pivot.values

    unique_codes = np.unique(img[~np.isnan(img)]).astype(int)
    legend_labels = [str(code) for code in unique_codes]
    code_to_idx = {code: i for i, code in enumerate(unique_codes)}
    base_cmap = plt.cm.get_cmap('tab20', len(unique_codes))

    # Prepare selection mask
    sel_idx = set()
    if selected_points is not None and len(selected_points) > 0:
        if isinstance(selected_points, (list, np.ndarray)):
            sel_idx = set(data.iloc[selected_points].index)
        else:
            sel_idx = set(selected_points.index)

    highlight_set = set(highlight_clusters)

    rgba_img = np.ones((*img.shape, 4))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = img[i, j]
            if np.isnan(val):
                rgba_img[i, j] = (1, 1, 1, 0)
                continue
            code = int(val)
            color = list(base_cmap(code_to_idx[code]))
            x = pivot.columns[j]
            y = pivot.index[i]
            idx = data[(data['x'] == x) & (data['y'] == y)].index
            cluster = data.loc[idx, 'cluster'].values[0] if len(idx) > 0 else None
            if len(idx) > 0 and idx[0] in sel_idx:
                color[-1] = 1.0
            elif cluster in highlight_set:
                color[-1] = 0.7
            else:
                color[-1] = 0.3
            rgba_img[i, j] = color

    plt.figure(figsize=(10, 5))
    plt.imshow(rgba_img, origin='lower', interpolation='none', aspect='auto')
    ax = plt.gca()
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    xtick_labels = [str(val) if i % 3 == 0 else '' for i, val in enumerate(pivot.columns)]
    ytick_labels = [str(val) if i % 3 == 0 else '' for i, val in enumerate(pivot.index)]
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(ytick_labels)
    ax.set_xticks(np.arange(-0.5, img.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, img.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)

    handles = [
        Patch(facecolor=base_cmap(i), edgecolor='none', label=legend_labels[i])
        for i in range(len(unique_codes))
    ]
    ncol = max(1, (len(unique_codes) + 9) // 10)
    plt.legend(
        handles=handles,
        title='Stratum',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        ncol=ncol
    )
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    ax.set_title(title, pad=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()



if __name__ == "__main__":
    # Create toy data
    np.random.seed(1)
    x = np.tile(np.arange(6), 6)
    y = np.repeat(np.arange(6), 6)
    cluster = (x + y) % 4  # 4 clusters_to_datapoints

    toy_data = pd.DataFrame({'x': x, 'y': y, 'cluster': cluster})

    # Plot using the function
    plot_cluster_grid(toy_data, 'cluster', 'Toy Cluster Grid Example')

    import numpy as np
    import pandas as pd
    from plot_cluster import plot_cluster_grid_highlight

    # Create toy data: 6x6 grid, 4 clusters_to_datapoints
    x = np.tile(np.arange(6), 6)
    y = np.repeat(np.arange(6), 6)
    cluster = (x + y) % 4  # 4 clusters_to_datapoints

    toy_data = pd.DataFrame({"x": x, "y": y, "cluster": cluster})

    # Randomly select clusters_to_datapoints to highlight
    np.random.seed(42)
    highlight_clusters = np.random.choice([0, 1, 2, 3], size=2, replace=False)

    # Plot and highlight
    plot_cluster_grid_highlight(
        data=toy_data,
        plot_variable="cluster",
        highlight_clusters=highlight_clusters,
        title="Toy Cluster Grid with Highlighted Clusters",
    )
