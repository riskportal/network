import contextlib
import random

import matplotlib.colors as colors
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
    foreground_color = "#ffffff"
    if background_color == "#ffffff":
        foreground_color = "#000000"

    domains = np.sort(annotation_matrix["domain"].unique())
    # domains = trimmed_domains_matrix.index.values

    # Define colors per domain
    domain2rgb = get_colors("hsv", len(domains))

    # Store domain info
    trimmed_domains_matrix["rgba"] = domain2rgb

    # Compute composite node colors
    node2nes = pd.DataFrame(
        data=neighborhood_enrichment_matrix,
        columns=[annotation_matrix.index.values, annotation_matrix["domain"]],
    )

    node2nes_binary = pd.DataFrame(
        data=neighborhood_binary_enrichment_matrix_below_alpha,
        columns=[annotation_matrix.index.values, annotation_matrix["domain"]],
    )
    node2domain_count = node2nes_binary.groupby(level="domain", axis=1).sum()
    node2all_domains_count = node2domain_count.sum(axis=1).to_numpy()[:, np.newaxis]

    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.matmul(node2domain_count.values, domain2rgb) / node2all_domains_count

    t = np.sum(c, axis=1)
    c[np.isnan(t) | np.isinf(t), :] = [0, 0, 0, 0]

    # Adjust brightness
    coeff_brightness = 0.1 / np.nanmean(np.ravel(c[:, :-1]))
    if coeff_brightness > 1:
        c = c * coeff_brightness
    c = np.clip(c, None, 1)

    # Sort nodes by their overall brightness
    ix = np.argsort(np.sum(c, axis=1))

    node_xy = get_node_coordinates(network)

    # Figure parameters
    num_plots = 2

    if show_each_domain:
        num_plots = num_plots + (len(domains) - 1)

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
    axes = axes.ravel()

    # First, plot the network
    ax = axes[0]
    ax = plot_network(network, ax=ax, background_color=background_color)

    print(node_xy)
    # Then, plot the composite network
    axes[1].scatter(node_xy[ix, 0], node_xy[ix, 1], c=c[ix], s=60, edgecolor=None)
    axes[1].set_aspect("equal")
    axes[1].set_facecolor(background_color)

    # Plot a circle around the network
    plot_network_contour(network, axes[1], background_color=background_color)

    # Plot the labels, if any
    if labels:
        plot_labels(labels, network, axes[1])

    if show_domain_ids:
        for domain in domains[domains > 0]:
            idx = domains_matrix["primary domain"] == domain
            centroid_x = np.nanmean(node_xy[idx, 0])
            centroid_y = np.nanmean(node_xy[idx, 1])
            axes[1].text(
                centroid_x,
                centroid_y,
                str(domain),
                fontdict={"size": 16, "color": foreground_color, "weight": "bold"},
            )

    # Then, plot each domain separately, if requested
    if show_each_domain:
        for domain in domains[domains > 0]:
            domain_color = np.reshape(domain2rgb[domain], (1, 4))

            alpha = node2nes.loc[:, domain].values
            alpha = alpha / max_log10_pvalue
            alpha[alpha > 1] = 1
            alpha = np.reshape(alpha, -1)

            c = np.repeat(domain_color, len(alpha), axis=0)
            # c[:, 3] = alpha

            idx = domains_matrix["primary domain"] == domain
            # ix = np.argsort(c)
            axes[1 + domain].scatter(
                node_xy[idx, 0], node_xy[idx, 1], c=c[idx], s=60, edgecolor=None
            )
            axes[1 + domain].set_aspect("equal")
            axes[1 + domain].set_facecolor(background_color)
            axes[1 + domain].set_title(
                "Domain %d\n%s" % (domain, trimmed_domains_matrix.loc[domain, "label"]),
                color=foreground_color,
            )
            plot_network_contour(network, axes[1 + domain], background_color=background_color)

            # Plot the labels, if any
            if labels:
                plot_labels(labels, network, axes[1 + domain])

    fig.set_facecolor(background_color)
    plt.savefig("./data/demo.png", facecolor=background_color)


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

    domains = np.sort(annotation_matrix["domain"].unique())
    # domains = trimmed_domains_matrix.index.values

    # Define colors per domain
    domain2rgb = get_colors("hsv", len(domains))

    # Store domain info
    trimmed_domains_matrix["rgba"] = domain2rgb

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


class MidpointRangeNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midrange=None, clip=False):
        self.midrange = midrange
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x = [self.vmin, self.midrange[0], self.midrange[1], self.midrange[2], self.vmax]
        y = [0, 0.25, 0.5, 0.75, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def get_colors(colormap="hsv", n=10):
    cmap = cm.get_cmap(colormap)

    # First color, always black
    rgbas = [(0, 0, 0, 1)]

    for c in np.arange(1, n):
        rgbas.append(cmap(c / n))
    rgba0 = rgbas.pop(0)
    random.shuffle(rgbas)
    rgbas = [rgba0] + rgbas
    # rgb = np.asarray(rgb)

    # # Randomize the other colors
    # np.random.shuffle(rgb[1:])

    # Set to RGBA standard
    # rgbas_scaled = []
    # for rgba in rgbas:
    #     rgbas_scaled.append(tuple(int(val * 255) if i != 3 else val for i, val in enumerate(rgba)))

    return rgbas
