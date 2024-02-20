import contextlib
import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm
from rich import print
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
    labels=[],
    show_each_domain=False,
    show_domain_ids=False,
    background_color="#000000",
):
    network = unfold_sphere_to_plane(network)
    foreground_color = "#ffffff" if background_color == "#000000" else "#000000"
    # Obtain unique domains and assign colors - do this prior to filtering to get some sweet color range
    domains = np.sort(annotation_matrix["domain"].unique())
    domain2rgb = get_colors("hsv", len(domains))
    # Omit bad group
    annotation_matrix = annotation_matrix[annotation_matrix["domain"] != 888888]
    domains_matrix = domains_matrix[domains_matrix["primary domain"] != 888888]
    trimmed_domains_matrix = trimmed_domains_matrix[trimmed_domains_matrix["id"] != 888888]
    # Create DataFrame mappings for node to enrichment score and binary presence
    neighborhood_enrichment_matrix = neighborhood_enrichment_matrix[
        :, annotation_matrix.index.values
    ]
    neighborhood_binary_enrichment_matrix_below_alpha = (
        neighborhood_binary_enrichment_matrix_below_alpha[:, annotation_matrix.index.values]
    )
    node2nes_binary = pd.DataFrame(
        data=neighborhood_binary_enrichment_matrix_below_alpha,
        columns=[annotation_matrix.index.values, annotation_matrix["domain"]],
    )
    node2domain_count = node2nes_binary.groupby(level="domain", axis=1).sum()
    # Order nodes by brightness for plotting
    composite_colors = get_composite_node_colors(domain2rgb, node2domain_count)

    # Identify rows where the fourth element is 1.0
    rows_with_alpha_one = composite_colors[:, 3] == 1
    # Assuming composite_colors is a NumPy array and rows_with_alpha_one is a boolean index or slice
    # Generate random weights - choose only one value as weight to synchronously change the color
    random_weights = np.random.uniform(0.80, 1.00, composite_colors[rows_with_alpha_one].shape[0])
    # For alpha - cuts the intensity of min -> max to half(min) -> max
    transformed_weights = 1.0 - (1.0 - random_weights) ** 2
    # Apply the random_weights to the first three columns of composite_colors
    # `[:, np.newaxis]` enables broadcasting across different dimensions
    composite_colors[rows_with_alpha_one, :3] *= random_weights[:, np.newaxis]
    # Apply the random weights to ALL elements of the selected rows
    composite_colors[rows_with_alpha_one, 3] *= transformed_weights

    node_xy = get_node_coordinates(network)
    # Begin trimming
    # Remove nodes that are very far away from their domain's hub
    node_xy_trimmed, composite_colors_trimmed, domains_matrix = remove_node_outliers_subclusters(
        node_xy, composite_colors, domains_matrix, std_dev_factor=2
    )
    # Plot node order from highest to lowest intensity
    node_order = get_refined_node_order(domains_matrix, composite_colors_trimmed)

    # Prepare figure layout
    num_plots = 2 + show_each_domain * len(domains)
    nrows = int(np.ceil(num_plots / 2))
    ncols = min(num_plots, 2)
    figsize = (10 * ncols, 10 * nrows)

    # ===== PLOTTING =====
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
    # NOTE: Ensure Alpha is always 1.0 - do this here to avoid tampering with argsort in line above
    # Example usage for one full rotation - plot images and export to ./png
    # rotate_and_project(node_xy_trimmed, node_order, composite_colors_trimmed, "rotate")
    # plot_composite_network_contours(network, node_xy_trimmed, annotation_matrix, domains_matrix, trimmed_domains_matrix, node_order)

    composite_colors_trimmed[:, 3] = np.where(composite_colors_trimmed[:, 3] > 0, 1.0, 0)
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
            node_order,
            composite_colors_trimmed,
            labels,
        )

    fig.set_facecolor(background_color)
    plt.savefig("./png/output/demo.png", facecolor=background_color, bbox_inches="tight")

    # # Prepare figure layout
    # show_each_domain = False
    # num_plots = 2 + show_each_domain * len(domains)
    # nrows = int(np.ceil(num_plots / 2))
    # ncols = min(num_plots, 2)
    # figsize = (10 * ncols, 10 * nrows)
    # fig, axes = plt.subplots(
    #     nrows=nrows,
    #     ncols=ncols,
    #     figsize=figsize,
    #     sharex=True,
    #     sharey=True,
    #     facecolor=background_color,
    # )
    # axes = axes.ravel()

    # plotter = NetworkPlotter(axes, background_color, foreground_color)
    # # NOTE: Ensure Alpha is always 1.0 - do this here to avoid tampering with argsort in line above
    # # Example usage for one full rotation - plot images and export to ./png
    # plotter.plot_main_network(network, node_xy_trimmed, node_order, composite_colors_trimmed)
    # fig.set_facecolor(background_color)
    # plt.savefig("./png/output/demo1.png", facecolor=background_color, bbox_inches="tight")


def get_composite_node_colors(domain2rgb, node2domain_count, random_seed=888):
    random.seed(random_seed)
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
        # BUG: THIS CONDITIONAL SHOULD NEVER BE FULFILLED AND WILL CAUSE PLOTTING ISSUES - THIS IS AN UPSTREAM
        # ISSUE THAT NEEDS TO BE RESOLVED
        if non_zero_domains > 0:
            composite_color = weighted_color_sum / non_zero_domains
        else:
            # Handle the case with no domain associations
            continue
        composite_colors[node_idx] = composite_color

    # Handle division by zero or other adjustments as necessary
    composite_colors = np.nan_to_num(composite_colors)

    return composite_colors


def remove_node_outliers_subclusters(node_xy, composite_colors, domains_matrix, std_dev_factor=2):
    """
    Remove outliers from node positions and corresponding composite colors,
    considering subclusters identified by the 'primary domain' in domains_matrix.
    Additionally, return the filtered domains matrix with the same indexing as its input.

    Parameters:
    - node_xy: Numpy array of node positions.
    - composite_colors: Numpy array of composite colors corresponding to node positions.
    - domains_matrix: DataFrame containing 'primary domain' column for subcluster identification.
    - std_dev_factor: Number of standard deviations for defining outliers within each subcluster.

    Returns:
    - Filtered node positions, composite colors, and domains matrix without outliers.
    """
    filtered_indices = []

    for domain in domains_matrix["primary domain"].unique():
        domain_indices = domains_matrix[domains_matrix["primary domain"] == domain].index
        subcluster_node_xy = node_xy[domain_indices]

        # Calculate the centroid and distances of points in the subcluster
        centroid = np.mean(subcluster_node_xy, axis=0)
        distances = np.linalg.norm(subcluster_node_xy - centroid, axis=1)
        distance_threshold = np.mean(distances) + std_dev_factor * np.std(distances)

        # Indices of non-outlier points within the current domain
        non_outlier_indices = domain_indices[distances <= distance_threshold]

        # Collect all non-outlier indices
        filtered_indices.extend(non_outlier_indices)

    # Use collected indices to filter node positions, composite colors, and the domains matrix
    filtered_node_xy = node_xy[filtered_indices]
    filtered_composite_colors = composite_colors[filtered_indices]
    filtered_domains_matrix = domains_matrix.loc[filtered_indices].copy()

    return filtered_node_xy, filtered_composite_colors, filtered_domains_matrix


class NetworkPlotter:
    def __init__(self, axes, background_color, foreground_color):
        self.axes = axes
        self.background_color = background_color
        self.foreground_color = foreground_color

    def plot_main_network(self, network, node_xy, node_order, composite_colors):
        # Plot the main network
        ax = self.axes[0]
        draw_network_perimeter(network, ax, self.background_color)
        plot_network(network, ax=ax, background_color=self.background_color)
        self.plot_composite_network(network, node_xy, node_order, composite_colors)

    def plot_composite_network(self, network, node_xy, node_order, composite_colors):
        ax = self.axes[1]
        draw_network_perimeter(network, ax, self.background_color)
        ax.scatter(
            node_xy[node_order, 0],
            node_xy[node_order, 1],
            c=composite_colors[node_order],
            s=60,
            edgecolor=None,
        )
        ax.set_aspect("equal")
        ax.set_facecolor(self.background_color)

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
        node_order,
        composite_colors,
        labels,
    ):
        if not show_each_domain:
            return
        ax_count = 2
        for domain in domains:
            # Identify indices where the 'primary domain' matches the current domain
            domains_to_filter = domains_matrix["primary domain"] == domain
            filtered_indices = np.where(domains_to_filter)[0]

            # Determine the order of these indices as per node_order
            # This ensures we only consider nodes in the current domain and follow the specified node_order
            ordered_indices = [idx for idx in node_order if idx in filtered_indices]

            # Check if there are any nodes to plot for this domain
            if not len(ordered_indices):
                continue

            # Convert to a NumPy array for easier indexing
            ordered_indices = np.array(ordered_indices)

            # Now, use these ordered indices to filter node_xy and composite_colors
            ordered_node_xy = node_xy[ordered_indices]
            ordered_composite_colors = composite_colors[ordered_indices]

            # Plotting
            ax = self.axes[ax_count]
            # First, draw the perimeter
            draw_network_perimeter(network, ax, self.background_color)
            ax.scatter(
                ordered_node_xy[:, 0],  # x-coordinates
                ordered_node_xy[:, 1],  # y-coordinates
                c=ordered_composite_colors,  # Colors
                s=60,  # Size of the marker
            )
            ax.set_aspect("equal")
            ax.set_facecolor(self.background_color)
            ax.set_title(
                f"Domain {ax_count}\n{trimmed_domains_matrix.loc[domain, 'label']}",
                color=self.foreground_color,
            )
            if labels:
                plot_labels(labels, network, ax)
            ax_count += 1


# def plot_composite_network_contours(
#     network,
#     node_xy,
#     annotation_matrix,
#     domains_matrix,
#     trimmed_domains_matrix,
#     node_order,
#     clabels=False,
#     background_color="#000000",
# ):
#     unique_domains = np.sort(annotation_matrix["domain"].unique())
#     # Get node coordinates

#     # Figure parameters
#     num_plots = 2

#     nrows = int(np.ceil(num_plots / 2))
#     ncols = np.min([num_plots, 2])
#     figsize = (10 * ncols, 10 * nrows)

#     [fig, axes] = plt.subplots(
#         nrows=nrows,
#         ncols=ncols,
#         figsize=figsize,
#         sharex=True,
#         sharey=True,
#         facecolor=background_color,
#     )

#     # Ensure axes is always a 1D array
#     axes = np.array(axes).ravel()
#     for ax in axes:
#         ax.set_facecolor(background_color)

#     # First, plot the network
#     ax = axes[1]
#     ax = plot_network(network, ax=ax, background_color=background_color)
#     # ax = axes[0]
#     # Then, plot the composite network as contours
#     for domain, terms in zip(trimmed_domains_matrix['id'], trimmed_domains_matrix['label']):
#         # This line throws key error for domain
#         # Identify indices where the 'primary domain' matches the current domain
#         # Identify indices where the 'primary domain' matches the current domain
#         domains_to_filter = domains_matrix["primary domain"] == domain
#         filtered_indices = np.where(domains_to_filter)[0]
#         # Determine the order of these indices as per node_order
#         # This ensures we only consider nodes in the current domain and follow the specified node_order
#         ordered_indices = [idx for idx in node_order if idx in filtered_indices]
#         # Check if there are any nodes to plot for this domain
#         if not len(ordered_indices):
#             continue
#         # Convert to a NumPy array for easier indexing
#         ordered_indices = np.array(ordered_indices)
#         draw_contours(axes[0], node_xy, ordered_indices)
#         # ax.scatter(
#         #     node_xy[ordered_indices][:, 0],  # x-coordinates
#         #     node_xy[ordered_indices][:, 1],  # y-coordinates
#         # )

#         if clabels:
#             C.levels = [domain + 1]
#             plt.clabel(C, C.levels, inline=True, fmt="%d", fontsize=16)
#             print("%d -- %s" % (domain + 1, domain))

#     fig.set_facecolor(background_color)
#     plt.savefig("test.png", facecolor=background_color)


# def draw_contours(ax, node_xy, ordered_indices, color='white', linewidths=1.5, levels=None):
#     """
#     Draw smooth contours around clusters of points.

#     Parameters:
#     - ax: The matplotlib axes to draw on.
#     - node_xy: A numpy array of node coordinates.
#     - ordered_indices: Indices of nodes that belong to a cluster.
#     - color: Color of the contour line.
#     - linewidths: Width of the contour lines.
#     - levels: Contour levels to draw.
#     """
#     if len(ordered_indices) == 0:
#         return  # No points to create a contour for.

#     pos = node_xy[ordered_indices, :]
#     kernel = gaussian_kde(pos.T)
#     xmin, xmax = np.min(pos[:, 0]), np.max(pos[:, 0])
#     ymin, ymax = np.min(pos[:, 1]), np.max(pos[:, 1])

#     # Create a grid of points where we want to evaluate the KDE
#     X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
#     positions = np.vstack([X.ravel(), Y.ravel()])
#     Z = np.reshape(kernel(positions).T, X.shape)

#     # If levels are not provided, determine them automatically
#     if levels is None:
#         Z_flattened_sorted = np.sort(Z.ravel())
#         # Take the value at the 90th percentile as the contour level
#         levels = [Z_flattened_sorted[int(0.9 * len(Z_flattened_sorted))]]

#     # Draw the contour around the cluster
#     contour = ax.contour(X, Y, Z, levels=levels, colors=color, linewidths=linewidths)

#     return contour


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


def draw_network_perimeter(graph, ax, background_color="#000000"):
    foreground_color = "#ffffff"
    if background_color == "#ffffff":
        foreground_color = "#000000"

    # Extract x and y coordinates from the graph nodes
    x = nx.get_node_attributes(graph, "x")
    y = nx.get_node_attributes(graph, "y")
    # Ensure positions do not contain NaN by filtering
    pos = {k: np.array([x[k], y[k]]) for k in x if not np.isnan(x[k]) and not np.isnan(y[k])}
    if len(pos) < 3:
        # ConvexHull requires at least 3 points to compute
        print("[red]ERROR: Could not compute graph...[/red]")
        return

    # Compute the convex hull to delineate the network
    hull = ConvexHull(np.array(list(pos.values())))
    # Extract the vertices of the hull to define the perimeter
    vertices_x = np.array([pos[v][0] for v in hull.vertices])
    vertices_y = np.array([pos[v][1] for v in hull.vertices])

    # Best fit a circle to these points
    def err(circle):
        xi, yi, ri = circle
        return sum((np.sqrt((vertices_x - xi) ** 2 + (vertices_y - yi) ** 2) - ri) ** 2)

    # Initial guess for the circle's center and radius based on mean values
    x0 = [
        np.mean(vertices_x),
        np.mean(vertices_y),
        np.mean(
            np.sqrt(
                (vertices_x - np.mean(vertices_x)) ** 2 + (vertices_y - np.mean(vertices_y)) ** 2
            )
        ),
    ]

    # Use fmin from scipy to minimize the error function and find the best circle
    xf, yf, rf = fmin(err, x0, disp=False)

    # Plot the circle with a dashed line
    circ = plt.Circle(
        (xf, yf),
        radius=rf,
        color=foreground_color,
        linewidth=2,
        alpha=0.8,
        fill=False,
        linestyle="dashed",
    )
    ax.add_patch(circ)

    ax.set_aspect("equal", "box")  # Set aspect ratio to be equal to make the circle look perfect
    ax.set_facecolor(background_color)  # Set the background color of the plot

    return xf, yf, rf


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


def get_refined_node_order(domains_matrix, composite_colors):
    # Ensure the input DataFrame is reindexed to start from 0 to align with composite_colors indices
    domains_matrix = domains_matrix.reset_index(drop=True)
    # Calculate the sum of composite_colors values for each entry
    composite_sums = np.sum(composite_colors, axis=1)
    # Create a DataFrame to map these sums with their corresponding indices
    composite_sums_df = pd.DataFrame(
        {"index": np.arange(len(composite_sums)), "composite_sum": composite_sums}
    )
    # Merge this DataFrame with the domains_matrix on the index to align composite sums with their domains
    merged_df = pd.merge(domains_matrix, composite_sums_df, left_index=True, right_on="index")
    # Group by 'primary domain' and calculate the necessary sorting metrics: total composite sum and group size
    group_metrics = (
        merged_df.groupby("primary domain")
        .agg(total_composite_sum=("composite_sum", "sum"), group_size=("composite_sum", "size"))
        .reset_index()
    )
    # Sort groups by group size (descending) and then by total composite sum (descending)
    sorted_groups = group_metrics.sort_values(
        by=["group_size", "total_composite_sum"], ascending=[True, True]
    )
    # Use the sorted group order to determine the order of rows in the merged DataFrame
    merged_df["group_order"] = pd.Categorical(
        merged_df["primary domain"], categories=sorted_groups["primary domain"], ordered=True
    )
    # Sort the merged DataFrame by this group order and then within groups by composite_sum (descending)
    final_sorted_df = merged_df.sort_values(
        by=["group_order", "composite_sum"], ascending=[False, True]
    )

    # Return the original indices of the rows in their new sorted order
    return final_sorted_df["index"].values


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
    # Evenly distribute the remaining colors: the -1 gives an excellent color spread
    color_positions = np.linspace(0, 1, num_colors_to_generate)
    random.shuffle(color_positions)
    # Generate colors based on positions
    rgbas = [cmap(pos) for pos in color_positions]

    return rgbas


def map_to_sphere(node_xy, composite_colors):
    # Normalize the coordinates between [0, 1]
    min_vals = np.min(node_xy, axis=0)
    max_vals = np.max(node_xy, axis=0)
    normalized_xy = (node_xy - min_vals) / (max_vals - min_vals)

    # Map normalized coordinates to theta and phi on a sphere
    theta = normalized_xy[:, 0] * np.pi * 2
    phi = normalized_xy[:, 1] * np.pi

    # Adjust radial distance based on alpha value (opacity) instead of color intensity
    # Extract alpha values from the last item in each color tuple
    alphas = composite_colors[:, -1]  # Assuming composite_colors is an array of RGBA values
    # Scale radial distances using alpha values; adjust this formula as needed
    r = alphas  # Direct use of alpha values; might adjust scaling based on your needs

    # Convert spherical coordinates to Cartesian coordinates for 3D sphere
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return np.vstack((x, y, z)).T


def unfold_sphere_to_plane(G):
    for node in G.nodes():
        if "z" in G.nodes[node]:
            x, y, z = G.nodes[node]["x"], G.nodes[node]["y"], G.nodes[node]["z"]
            # Calculate theta and phi from Cartesian coordinates
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)
            phi = np.arccos(z / r)

            # Adjust theta and phi to unfold the sphere
            unfolded_x = (theta + np.pi) / (2 * np.pi)  # Shift and normalize theta to [0, 1]
            unfolded_y = (np.pi - phi) / np.pi  # Reflect phi and normalize to [0, 1]
            unfolded_x = unfolded_x + 0.5 if unfolded_x < 0.5 else unfolded_x - 0.5

            G.nodes[node]["x"] = unfolded_x
            G.nodes[node]["y"] = -unfolded_y

            # Remove the 'z' coordinate
            del G.nodes[node]["z"]

    return G


def rotate_and_project(node_xy, node_order, composite_colors, filename_prefix):
    num_angles = 240
    angles = np.linspace(0, 360, num_angles, endpoint=False)

    # Map initial 2D coordinates onto a sphere for the initial 3D coordinates
    nodes_3d = map_to_sphere(node_xy, composite_colors)

    # Initial tilt downwards: 180 degrees around the X-axis to bring the top face to the bottom
    initial_tilt_angle_radians = np.radians(180)
    initial_tilt_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(initial_tilt_angle_radians), -np.sin(initial_tilt_angle_radians)],
            [0, np.sin(initial_tilt_angle_radians), np.cos(initial_tilt_angle_radians)],
        ]
    )
    nodes_3d = np.dot(nodes_3d, initial_tilt_matrix)

    plotting_range = np.max(np.abs(nodes_3d)) * 1.1

    for angle in angles:
        angle_radians = np.radians(angle)
        rotation_matrix_y = np.array(
            [
                [np.cos(angle_radians), 0, np.sin(angle_radians)],
                [0, 1, 0],
                [-np.sin(angle_radians), 0, np.cos(angle_radians)],
            ]
        )

        rotated_nodes_3d = np.dot(nodes_3d, rotation_matrix_y)

        depth_order = np.argsort(rotated_nodes_3d[:, 2])
        ordered_colors = [composite_colors[i] for i in depth_order]
        rotated_xy = rotated_nodes_3d[:, :2]
        plt.figure(figsize=(8, 8))
        plt.scatter(
            rotated_xy[depth_order, 0],
            rotated_xy[depth_order, 1],
            c=ordered_colors,
            s=60,
            edgecolor="none",
            alpha=1,
        )
        plt.gca().set_aspect("equal", "box")
        plt.gca().set_facecolor("black")
        plt.xlim(-plotting_range, plotting_range)
        plt.ylim(-plotting_range, plotting_range)
        plt.axis("off")

        filename = f"./png/for_gif/{filename_prefix}_{int(angle):03d}.png"
        plt.savefig(filename, bbox_inches="tight", facecolor="black")
        plt.close()
