import contextlib
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.spatial import ConvexHull
from scipy.optimize import fmin
from scipy.stats import gaussian_kde


def plot_composite_network(
    network,
    annotation_matrix,
    domains_matrix,
    trimmed_domains_matrix,
    neighborhood_enrichment_matrix,
    neighborhood_binary_enrichment_matrix_below_alpha,
    max_log10_pvalue,
    labels=[],
    show_each_domain=False,
    show_domain_ids=False,
    background_color="#000000",
):
    foreground_color = "#ffffff" if background_color == "#000000" else "#000000"

    # Obtain unique domains and assign colors
    domains = np.sort(annotation_matrix["domain"].unique())
    domain2rgb = get_colors("hsv", len(domains))
    # Create DataFrame mappings for node to enrichment score and binary presence
    neighborhood_enrichment_matrix = neighborhood_enrichment_matrix[
        :, annotation_matrix.index.values
    ]
    node2nes = pd.DataFrame(
        data=neighborhood_enrichment_matrix,
        columns=[annotation_matrix.index.values, annotation_matrix["domain"]],
    )
    neighborhood_binary_enrichment_matrix_below_alpha = (
        neighborhood_binary_enrichment_matrix_below_alpha[:, annotation_matrix.index.values]
    )
    node2nes_binary = pd.DataFrame(
        data=neighborhood_binary_enrichment_matrix_below_alpha,
        columns=[annotation_matrix.index.values, annotation_matrix["domain"]],
    )
    node2domain_count = node2nes_binary.groupby(level="domain", axis=1).sum()
    composite_colors = get_composite_node_colors(domain2rgb, node2domain_count)
    # Omit bad group
    domains_matrix = domains_matrix[domains_matrix["primary domain"] != 888888]
    trimmed_indices = domains_matrix.index
    # Order nodes by brightness for plotting
    node_xy = get_node_coordinates(network)
    node_xy_trimmed = node_xy[trimmed_indices]
    composite_colors_trimmed = composite_colors[trimmed_indices]

    # Prepare figure layout
    num_plots = 2 + show_each_domain * len(domains)
    nrows = int(np.ceil(num_plots / 2))
    ncols = min(num_plots, 2)
    figsize = (10 * ncols, 10 * nrows)

    # Create the subplots dynamically based on the number of plots needed
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        facecolor=background_color,
    )
    axes = axes.ravel()

    plotter = NetworkPlotter(axes, background_color, foreground_color)
    node_order = np.argsort(np.sum(composite_colors_trimmed, axis=1))
    # NOTE: Ensure Alpha is always 1.0 - do this here to avoid tampering with argsort in line above
    composite_colors_trimmed[:, 3] = 1.0
    plotter.plot_main_network(network, node_xy_trimmed, node_order, composite_colors_trimmed)

    if show_domain_ids:
        plotter.plot_domain_ids(domains, domains_matrix, node_xy)

    if show_each_domain:
        plotter.plot_each_domain(
            network,
            show_each_domain,
            domains,
            domains_matrix,
            trimmed_domains_matrix,
            node_xy_trimmed,
            node2nes,
            composite_colors_trimmed,
            max_log10_pvalue,
            labels,
        )

    fig.set_facecolor(background_color)
    plt.savefig("./data/demo.png", facecolor=background_color, bbox_inches="tight")


def get_composite_node_colors(domain2rgb, node2domain_count):
    # Ensure domain2rgb is a numpy array. If it's not, convert it:
    # If domain2rgb is a list of tuples/lists, convert it to a numpy array
    if not isinstance(domain2rgb, np.ndarray):
        domain2rgb = np.array(domain2rgb)

    # Placeholder for the output composite colors, one row per node
    composite_colors = np.zeros((node2domain_count.shape[0], 4))  # Assuming RGBA
    # Iterate over each node to compute its composite color
    for node_idx in range(node2domain_count.shape[0]):
        # Extract the domain counts for this node
        domain_counts = node2domain_count.values[node_idx, :]
        # Handle division by zero by checking if the maximum count is greater than 0
        max_count = np.max(domain_counts)
        if max_count > 0:
            normalized_counts = domain_counts / max_count
        else:
            # Handle the case where max_count is 0 to avoid division by zero
            normalized_counts = domain_counts
        # Compute the weighted color for each domain and sum them
        weighted_color_sum = np.zeros(4)
        for domain_idx, count in enumerate(normalized_counts):
            color = domain2rgb[domain_idx]  # Ensure color is an array
            weighted_color = color * count  # Element-wise multiplication
            weighted_color_sum += weighted_color
        # Normalize the weighted sum of colors by the number of domains with non-zero counts
        non_zero_domains = np.count_nonzero(normalized_counts)
        if non_zero_domains > 0:
            composite_color = weighted_color_sum / non_zero_domains
        else:
            # Handle the case with no domain associations
            composite_color = np.array([0, 0, 0, 1])  # Default color, e.g., transparent or black
        composite_colors[node_idx] = composite_color

    # Handle division by zero or other adjustments as necessary
    composite_colors = np.nan_to_num(composite_colors)

    return composite_colors


class NetworkPlotter:
    def __init__(self, axes, background_color, foreground_color):
        self.axes = axes
        self.background_color = background_color
        self.foreground_color = foreground_color

    def plot_main_network(self, network, node_xy, node_order, composite_colors):
        # Plot the main network
        plot_network(network, ax=self.axes[0], background_color=self.background_color)
        self.plot_composite_network(network, node_xy, node_order, composite_colors)

    def plot_composite_network(self, network, node_xy, node_order, composite_colors):
        ax = self.axes[1]
        ax.scatter(
            node_xy[node_order, 0],
            node_xy[node_order, 1],
            c=composite_colors[node_order],
            s=60,
            edgecolor=None,
        )
        ax.set_aspect("equal")
        ax.set_facecolor(self.background_color)
        plot_network_contour(network, ax, self.background_color)

    def plot_domain_ids(self, domains, domains_matrix, node_xy):
        ax = self.axes[1]
        for domain in domains:
            domain_indices = domains_matrix["primary domain"] == domain
            centroid_x, centroid_y = np.nanmean(node_xy[domain_indices, :], axis=0)
            ax.text(
                centroid_x,
                centroid_y,
                str(domain),
                fontdict={"size": 16, "color": self.foreground_color, "weight": "bold"},
            )

    def plot_each_domain(
        self,
        network,
        show_each_domain,
        domains,
        domains_matrix,
        trimmed_domains_matrix,
        node_xy,
        node2nes,
        composite_colors,
        max_log10_pvalue,
        labels,
    ):
        if not show_each_domain:
            return
        ax_count = 2
        for domain in domains:
            with contextlib.suppress(KeyError):
                alpha = node2nes.loc[:, domain].values / max_log10_pvalue
                alpha = np.clip(alpha, 0, 1)  # Clip alpha values between 0 and 1
                alpha = np.reshape(alpha, -1)
                # composite_colors[:, 3] = alpha  # Super dim
                domains_to_filter = domains_matrix["primary domain"] == domain
                # Empty matrices
                if not node_xy[domains_to_filter, 0].shape[0]:
                    continue
                # Now begin to plot
                ax = self.axes[ax_count]
                ax.scatter(
                    node_xy[domains_to_filter, 0],
                    node_xy[domains_to_filter, 1],
                    c=composite_colors[domains_to_filter],
                    s=60,
                )
                ax.set_aspect("equal")
                ax.set_facecolor(self.background_color)
                ax.set_title(
                    f"Domain {ax_count}\n{trimmed_domains_matrix.loc[domain, 'label']}",
                    color=self.foreground_color,
                )
                plot_network_contour(network, ax, self.background_color)
                if labels:
                    plot_labels(labels, network, ax)
                ax_count += 1


def remove_outliers(data_dict, z_score_threshold=3):
    values = np.array(list(data_dict.values()))
    # Calculate mean and standard deviation
    mean = np.mean(values)
    std_dev = np.std(values)

    # Function to calculate Z-score
    def calculate_z_score(value):
        return (value - mean) / std_dev

    # Identify and remove outliers
    cleaned_dict = {}
    for key, value in data_dict.items():
        z_score = calculate_z_score(value)
        if abs(z_score) <= z_score_threshold:
            cleaned_dict[key] = value

    return cleaned_dict


def plot_composite_network_contours(
    network,
    annotation_matrix,
    domains_matrix,
    trimmed_domains_matrix,
    clabels=False,
    background_color="#000000",
):
    foreground_color = "#ffffff"
    if background_color == "#ffffff":
        foreground_color = "#000000"
    unique_domains = np.sort(annotation_matrix["domain"].unique())
    # Define colors per domain
    unique_domain_colors = get_colors("viridis", len(unique_domains))
    # Get node coordinates
    node_xy = get_node_coordinates(network)

    # Figure parameters
    num_plots = 2

    nrows = int(np.ceil(num_plots / 2))
    ncols = np.min([num_plots, 2])
    figsize = (10 * ncols, 10 * nrows)

    [fig, axes] = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        facecolor=background_color,
    )

    # Ensure axes is always a 1D array
    axes = np.array(axes).ravel()

    # First, plot the network
    ax = axes[1]
    ax = plot_network(network, ax=ax, background_color=background_color)

    # Then, plot the composite network as contours
    for n_domain, domain in enumerate(trimmed_domains_matrix["label"].values):
        with contextlib.suppress(KeyError):
            # This line throws key error for domain
            nodes_indices = domains_matrix.loc[domains_matrix.loc[:, n_domain] > 0,].index.values
            pos3 = node_xy[nodes_indices, :]
            kernel = gaussian_kde(pos3.T)
            [X, Y] = np.mgrid[
                np.min(pos3[:, 0]) : np.max(pos3[:, 0]) : 100j,
                np.min(pos3[:, 1]) : np.max(pos3[:, 1]) : 100j,
            ]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions).T, X.shape)

            # Use ax instead of axes[1], and specify the subplot position
            C = ax[0].contour(
                X, Y, Z, [1e-6], colors=[trimmed_domains_matrix.loc[n_domain, "rgba"]], alpha=1
            )

            if clabels:
                C.levels = [n_domain + 1]
                plt.clabel(C, C.levels, inline=True, fmt="%d", fontsize=16)
                print("%d -- %s" % (n_domain + 1, domain))

    fig.set_facecolor(background_color)
    plt.savefig("demo.png", facecolor=background_color)


def plot_network(G, ax=None, background_color="#000000"):
    foreground_color = "#ffffff"
    if background_color == "#ffffff":
        foreground_color = "#000000"

    node_xy = get_node_coordinates(G)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(20, 10), facecolor=background_color, edgecolor=foreground_color
        )
        fig.set_facecolor(background_color)

    # Randomly sample a fraction of the edges (when network is too big)
    edges = tuple(G.edges())
    if len(edges) > 30000:
        edges = random.sample(edges, int(len(edges) * 0.1))

    nx.draw(
        G,
        ax=ax,
        pos=node_xy,
        edgelist=edges,
        node_color=foreground_color,
        edge_color=foreground_color,
        node_size=10,
        width=1,
        alpha=0.2,
    )

    ax.set_aspect("equal")
    ax.set_facecolor(background_color)

    ax.grid(False)
    ax.invert_yaxis()
    ax.margins(0.1, 0.1)

    ax.set_title("Network", color=foreground_color)

    plt.axis("off")

    try:
        fig.set_facecolor(background_color)
    except NameError:
        pass

    return ax


def plot_network_contour(graph, ax, background_color="#000000"):
    foreground_color = "#ffffff"
    if background_color == "#ffffff":
        foreground_color = "#000000"

    x = dict(graph.nodes.data("x"))
    y = dict(graph.nodes.data("y"))

    ds = [x, y]
    pos = {}
    for k in x:
        pos[k] = np.array([d[k] for d in ds])

    # Compute the convex hull to delineate the network
    hull = ConvexHull(np.vstack(list(pos.values())))

    vertices_x = [pos.get(v)[0] for v in hull.vertices]
    vertices_y = [pos.get(v)[1] for v in hull.vertices]

    vertices_x = np.array(vertices_x)
    vertices_y = np.array(vertices_y)

    # Find center of mass and radius to approximate the hull with a circle
    xm = np.nanmean(vertices_x)
    ym = np.nanmean(vertices_y)

    rm = np.nanmean(np.sqrt((vertices_x - xm) ** 2 + (vertices_y - ym) ** 2))

    # Best fit a circle to these points
    def err(x0):
        [w, v, r] = x0
        pts = [np.linalg.norm([x - w, y - v]) - r for x, y in zip(vertices_x, vertices_y)]
        return (np.array(pts) ** 2).sum()

    [xf, yf, rf] = fmin(err, [xm, ym, rm], disp=False)

    circ = plt.Circle((xf, yf), radius=rf * 1.01, color=foreground_color, linewidth=1, fill=False)
    ax.add_patch(circ)

    return xf, yf, rf


def plot_costanzo2016_network_annotations(
    graph, ax, path_to_data, colors=True, clabels=False, background_color="#000000"
):
    foreground_color = "#ffffff"
    if background_color == "#ffffff":
        foreground_color = "#000000"

    path_to_network_annotations = (
        "other/Data File S5_SAFE analysis_Gene cluster identity and functional enrichments.xlsx"
    )
    filename = os.path.join(path_to_data, path_to_network_annotations)

    costanzo2016 = pd.read_excel(filename, sheet_name="Global net. cluster gene list")
    processes = costanzo2016["Global Similarity Network Region name"].unique()
    processes = processes[pd.notnull(processes)]

    process_colors = pd.read_csv(
        os.path.join(path_to_data, "other/costanzo_2016_colors.txt"), sep="\t"
    )
    if colors:
        process_colors = process_colors[["R", "G", "B"]].values / 256
    else:
        if foreground_color == "#ffffff":
            process_colors = np.ones((process_colors.shape[0], 3))
        else:
            process_colors = np.zeros((process_colors.shape[0], 3))

    labels = nx.get_node_attributes(graph, "label")
    labels_dict = {k: v for v, k in labels.items()}

    x = list(dict(graph.nodes.data("x")).values())
    y = list(dict(graph.nodes.data("y")).values())

    pos = {}
    for idx, k in enumerate(x):
        pos[idx] = np.array([x[idx], y[idx]])

    for n_process, process in enumerate(processes):
        nodes = costanzo2016.loc[
            costanzo2016["Global Similarity Network Region name"] == process, "Gene/Allele"
        ]
        nodes_indices = [labels_dict[node] for node in nodes if node in labels_dict.keys()]

        pos3 = {idx: pos[node_index] for idx, node_index in enumerate(nodes_indices)}
        pos3 = np.vstack(list(pos3.values()))

        kernel = gaussian_kde(pos3.T)
        [X, Y] = np.mgrid[np.min(x) : np.max(x) : 100j, np.min(y) : np.max(y) : 100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)

        C = ax.contour(X, Y, Z, [1e-6], colors=[tuple(process_colors[n_process, :])], alpha=1)

        if clabels:
            C.levels = [n_process + 1]
            plt.clabel(C, C.levels, inline=True, fmt="%d", fontsize=16)
            print("%d -- %s" % (n_process + 1, process))


def plot_labels(labels, graph, ax):
    node_labels = nx.get_node_attributes(graph, "label")
    node_labels_dict = {k: v for v, k in node_labels.items()}

    x = list(dict(graph.nodes.data("x")).values())
    y = list(dict(graph.nodes.data("y")).values())

    # x_offset = (np.nanmax(x) - np.nanmin(x))*0.01

    idx = [node_labels_dict[x] for x in labels if x in node_labels_dict.keys()]
    labels_idx = [x for x in labels if x in node_labels_dict.keys()]
    x_idx = [x[i] for i in idx]
    y_idx = [y[i] for i in idx]

    # ax.plot(x_idx, y_idx, 'r*')
    for i in np.arange(len(idx)):
        ax.text(
            x_idx[i],
            y_idx[i],
            labels_idx[i],
            fontdict={"color": "white", "size": 14, "weight": "bold"},
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 3},
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Print out labels not found
    labels_missing = [x for x in labels if x not in node_labels_dict.keys()]
    if labels_missing:
        labels_missing_str = ", ".join(labels_missing)
        print("These labels are missing from the network (case sensitive): %s" % labels_missing_str)


def get_node_coordinates(graph):
    x = dict(graph.nodes.data("x"))
    y = dict(graph.nodes.data("y"))

    ds = [x, y]
    pos = {}
    for k in x:
        pos[k] = np.array([d[k] for d in ds])

    node_xy = np.vstack(list(pos.values()))

    return node_xy


def get_colors(colormap="plasma", num_colors_to_generate=10, random_seed=888):
    # Updated colormap keywords with diverse colormaps and removed less distinct ones
    colormap_keywords = {
        "viridis",
        "plasma",
        "inferno",
        "hsv",
        "magma",
        "cividis",
        "rainbow",
        "jet",
        # Adding more diverse colormaps
        "twilight",
        "nipy_spectral",
        "ocean",
        "gist_earth",
        "terrain",
        "gist_ncar",
        "Spectral",
        "coolwarm",
    }
    random.seed(random_seed)
    cmap = cm.get_cmap("hsv")
    # First color, always black
    # Evenly distribute the remaining colors
    color_positions = np.linspace(0, 1, num_colors_to_generate - 1)
    random.shuffle(color_positions)
    # Generate colors based on positions
    rgbas = [cmap(pos) for pos in color_positions]

    return [(0, 0, 0, 1)] + rgbas
