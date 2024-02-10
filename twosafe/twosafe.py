import os
import sys
import textwrap
import argparse
import pickle
import time
import re

import pandas as pd

# Necessary check to make sure code runs both in Jupyter and in command line
if "matplotlib" not in sys.modules:
    import matplotlib

    matplotlib.use("Agg")

# import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# import multiprocessing as mp

from matplotlib.colors import LinearSegmentedColormap
from itertools import compress
from scipy.cluster.hierarchy import linkage, fcluster


from .network.io import load_cys_network, load_network_annotation
from .network.neighborhoods import get_network_neighborhoods, define_top_attributes
from .network.stats import compute_pvalues_by_randomization
from .config import read_default_config, validate_config


class SAFE:
    """SAFE"""

    def __init__(self, network_filepath="", annotation_filepath="", **kwargs):
        """
        Initiate a SAFE instance and define the main settings for analysis.
        The settings are automatically extracted from the specified (or default) INI configuration file.
        Alternatively, each setting can be changed manually after initiation.

        :param path_to_ini_file (str): Path to the configuration file. If not specified, safe_default.ini will be used.
        :param verbose (bool): Defines whether or not intermediate output will be printed out.

        """
        user_input = {
            **kwargs,
            "network_filepath": network_filepath,
            "annotation_filepath": annotation_filepath,
        }
        self.config = validate_config({**read_default_config(), **user_input})
        # TODO: What if the API is something like twosafe.load_cytoscape_network --> SAFE object
        self.network = self.load_cytoscape_network()
        self.annotation_map = self.load_network_annotation(self.network)
        self.neighborhoods = self.load_neighborhoods(self.network)
        self.neighborhood_enrichment_map = self.get_pvalues(self.neighborhoods, self.annotation_map)
        print(
            self.define_top_attributes(
                self.network, self.annotation_map, self.neighborhood_enrichment_map
            )
        )

        self.default_config = None
        self.path_to_safe_data = None
        self.path_to_network_file = None
        self.view_name = None
        self.path_to_attribute_file = None

        self.graph = None
        self.node_key_attribute = "label_orf"

        self.attributes_matrix = None
        self.nodes = None
        self.node2attribute = None
        self.num_nodes_per_attribute = None
        self.attribute_sign = "both"

        self.node_distance_metric = "shortpath_weighted_layout"
        self.neighborhood_radius_type = None
        self.neighborhood_radius = None

        self.background = "attribute_file"
        self.num_permutations = 1000
        self.multiple_testing = False
        self.neighborhood_score_type = "sum"
        self.enrichment_type = "auto"
        self.enrichment_threshold = 0.05
        self.enrichment_max_log10 = 16
        self.attribute_enrichment_min_size = 10

        self.neighborhoods = None

        self.ns = None
        neg_pvals = None
        pos_pvals = None
        self.nes = None
        self.nes_threshold = None
        self.nes_binary = None

        self.attribute_unimodality_metric = "connectivity"
        self.attribute_distance_metric = "jaccard"
        self.attribute_distance_threshold = 0.75

        self.domains = None
        self.node2domain = None

        # Output
        self.output_dir = ""

    def load_cytoscape_network(self, *args, **kwargs):
        network_filepath = self.config["network_filepath"]
        return load_cys_network(network_filepath, *args, **kwargs)
        # except KeyError as e:
        #     raise KeyError(
        #         f"{network_file_extension} is not a valid network file extension."
        #         f" Choose from the following: {', '.join(self.network_options)}"
        #     ) from e

    # def get_network_nodes(self, network):
    #     annotation_id_colname = self.config["annotation_id_colname"]
    #     node_attributes = nx.get_node_attributes(network, annotation_id_colname)
    #     if not bool(node_attributes):
    #         raise AttributeError(
    #             f"{self.config['annotation_id_colname']} is not a valid column name for annotation keys."
    #             f"Consider setting `annotation_id_colname` to one of the following {', '.join(network.nodes[0].keys())}"
    #         )
    #     nx.set_node_attributes(network, annotation_id_colname, name="key")
    #     label_list = nx.get_node_attributes(network, "label")
    #     return pd.DataFrame(
    #         {
    #             "id": list(label_list.keys()),
    #             "key": list(node_attributes.values()),
    #             "label": list(label_list.values()),
    #         }
    #     )

    def load_network_annotation(self, network):
        annotation = load_network_annotation(
            network, self.config["annotation_filepath"], self.config["annotation_id_colname"]
        )
        return annotation

    def load_neighborhoods(self, network):
        neighborhoods = get_network_neighborhoods(
            network, self.config["neighborhood_distance_metric"], self.config["neighborhood_radius"]
        )
        return neighborhoods

    def get_pvalues(self, neighborhoods, annotation):
        annotation_matrix = annotation["annotation_matrix"]
        neighborhood_enrichment_map = compute_pvalues_by_randomization(
            neighborhoods,
            annotation_matrix,
            self.config["neighborhood_score_metric"],
            self.config["network_enrichment_direction"],
            self.config["enrichment_alpha_cutoff"],
            num_permutations=2,
            max_workers=1,
            random_seed=888,
            multiple_testing=False,
        )
        return neighborhood_enrichment_map

    def define_top_attributes(self, network, annotation_map, neighborhoods_map):
        ordered_column_annotations = annotation_map["ordered_column_annotations"]
        neighborhood_enrichment_sums = neighborhoods_map["neighborhood_enrichment_sums"]
        neighborhood_binary_enrichment_matrix_below_alpha = neighborhoods_map[
            "neighborhood_binary_enrichment_matrix_below_alpha"
        ]
        return define_top_attributes(
            network,
            ordered_column_annotations,
            neighborhood_enrichment_sums,
            neighborhood_binary_enrichment_matrix_below_alpha,
            self.config["min_annotation_size"],
            self.config["unimodality_type"],
        )

    def define_domains(self, **kwargs):
        # Overwriting global settings, if necessary
        if "attribute_distance_threshold" in kwargs:
            self.attribute_distance_threshold = kwargs["attribute_distance_threshold"]

        # Make sure that the settings are still valid
        self.validate_config()

        m = self.nes_binary[:, self.attributes_matrix["top"]].T
        Z = linkage(m, method="average", metric=self.attribute_distance_metric)
        max_d = np.max(Z[:, 2] * self.attribute_distance_threshold)
        domains = fcluster(Z, max_d, criterion="distance")

        self.attributes_matrix["domain"] = 0
        self.attributes_matrix.loc[self.attributes_matrix["top"], "domain"] = domains

        # Assign nodes to domains
        node2nes = pd.DataFrame(
            data=self.nes,
            columns=[self.attributes_matrix.index.values, self.attributes_matrix["domain"]],
        )
        node2nes_binary = pd.DataFrame(
            data=self.nes_binary,
            columns=[self.attributes_matrix.index.values, self.attributes_matrix["domain"]],
        )

        # # A node belongs to the domain that contains the attribute
        # for which the node has the highest enrichment
        # self.node2domain = node2nes.groupby(level='domain', axis=1).max()
        # t_max = self.node2domain.loc[:, 1:].max(axis=1)
        # t_idxmax = self.node2domain.loc[:, 1:].idxmax(axis=1)
        # t_idxmax[t_max < -np.log10(self.enrichment_threshold)] = 0

        # A node belongs to the domain that contains the highest number of attributes
        # for which the nodes is significantly enriched
        self.node2domain = node2nes_binary.groupby(level="domain", axis=1).sum()
        t_max = self.node2domain.loc[:, 1:].max(axis=1)
        t_idxmax = self.node2domain.loc[:, 1:].idxmax(axis=1)
        t_idxmax[t_max == 0] = 0

        self.node2domain["primary_domain"] = t_idxmax

        # Get the max NES for the primary domain
        o = node2nes.groupby(level="domain", axis=1).max()
        i = pd.Series(t_idxmax)
        self.node2domain["primary_nes"] = o.lookup(i.index, i.values)

        if self.verbose:
            num_domains = len(np.unique(domains))
            num_attributes_per_domain = (
                self.attributes_matrix.loc[self.attributes_matrix["domain"] > 0]
                .groupby("domain")["id"]
                .count()
            )
            min_num_attributes = num_attributes_per_domain.min()
            max_num_attributes = num_attributes_per_domain.max()
            print(
                "Number of domains: %d (containing %d-%d attributes)"
                % (num_domains, min_num_attributes, max_num_attributes)
            )

    def trim_domains(self, **kwargs):
        # Remove domains that are the top choice for less than a certain number of neighborhoods
        domain_counts = np.zeros(len(self.attributes_matrix["domain"].unique())).astype(int)
        t = self.node2domain.groupby("primary_domain")["primary_domain"].count()
        domain_counts[t.index] = t.values
        to_remove = np.flatnonzero(domain_counts < self.attribute_enrichment_min_size)

        self.attributes_matrix.loc[self.attributes_matrix["domain"].isin(to_remove), "domain"] = 0

        idx = self.node2domain["primary_domain"].isin(to_remove)
        self.node2domain.loc[idx, ["primary_domain", "primary_nes"]] = 0

        # Rename the domains (simple renumber)
        a = np.sort(self.attributes_matrix["domain"].unique())
        b = np.arange(len(a))
        renumber_dict = dict(zip(a, b))

        self.attributes_matrix["domain"] = [
            renumber_dict[k] for k in self.attributes_matrix["domain"]
        ]
        self.node2domain["primary_domain"] = [
            renumber_dict[k] for k in self.node2domain["primary_domain"]
        ]
        self.node2domain.drop(columns=to_remove)

        # Make labels for each domain
        domains = np.sort(self.attributes_matrix["domain"].unique())
        domains_labels = self.attributes_matrix.groupby("domain")["name"].apply(chop_and_filter)
        self.domains = pd.DataFrame(data={"id": domains, "label": domains_labels})
        self.domains.set_index("id", drop=False)

        if self.verbose:
            print(
                "Removed %d domains because they were the top choice for less than %d neighborhoods."
                % (len(to_remove), self.attribute_enrichment_min_size)
            )

    def plot_network(self, background_color="#000000"):
        plot_network(self.graph, background_color=background_color)

    def plot_composite_network_contours(
        self, save_fig=None, clabels=False, background_color="#000000"
    ):
        foreground_color = "#ffffff"
        if background_color == "#ffffff":
            foreground_color = "#000000"

        domains = np.sort(self.attributes_matrix["domain"].unique())
        # domains = self.domains.index.values

        # Define colors per domain
        domain2rgb = get_colors("hsv", len(domains))

        # Store domain info
        self.domains["rgba"] = domain2rgb.tolist()

        # Get node coordinates
        node_xy = get_node_coordinates(self.graph)

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
        axes = axes.ravel()

        # First, plot the network
        ax = axes[0]
        ax = plot_network(self.graph, ax=ax, background_color=background_color)

        # Then, plot the composite network as contours

        for n_domain, domain in enumerate(self.domains["label"].values):
            nodes_indices = self.node2domain.loc[
                self.node2domain.loc[:, n_domain] > 0,
            ].index.values
            pos3 = node_xy[nodes_indices, :]

            kernel = gaussian_kde(pos3.T)
            [X, Y] = np.mgrid[
                np.min(pos3[:, 0]) : np.max(pos3[:, 0]) : 100j,
                np.min(pos3[:, 1]) : np.max(pos3[:, 1]) : 100j,
            ]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions).T, X.shape)

            C = ax[1].contour(X, Y, Z, [1e-6], colors=self.domains.loc[n_domain, "rgba"], alpha=1)

            if clabels:
                C.levels = [n_domain + 1]
                plt.clabel(C, C.levels, inline=True, fmt="%d", fontsize=16)
                print("%d -- %s" % (n_domain + 1, domain))

        fig.set_facecolor(background_color)

        if save_fig:
            path_to_fig = save_fig
            print("Output path: %s" % path_to_fig)
            plt.savefig(path_to_fig, facecolor=background_color)

    def plot_composite_network(
        self,
        show_each_domain=False,
        show_domain_ids=True,
        save_fig=None,
        labels=[],
        background_color="#000000",
    ):
        foreground_color = "#ffffff"
        if background_color == "#ffffff":
            foreground_color = "#000000"

        domains = np.sort(self.attributes_matrix["domain"].unique())
        # domains = self.domains.index.values

        # Define colors per domain
        domain2rgb = get_colors("hsv", len(domains))

        # Store domain info
        self.domains["rgba"] = domain2rgb.tolist()

        # Compute composite node colors
        node2nes = pd.DataFrame(
            data=self.nes,
            columns=[self.attributes_matrix.index.values, self.attributes_matrix["domain"]],
        )

        node2nes_binary = pd.DataFrame(
            data=self.nes_binary,
            columns=[self.attributes_matrix.index.values, self.attributes_matrix["domain"]],
        )
        node2domain_count = node2nes_binary.groupby(level="domain", axis=1).sum()
        node2all_domains_count = node2domain_count.sum(axis=1)[:, np.newaxis]

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

        node_xy = get_node_coordinates(self.graph)

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
        ax = plot_network(self.graph, ax=ax, background_color=background_color)

        # Then, plot the composite network
        axes[1].scatter(node_xy[ix, 0], node_xy[ix, 1], c=c[ix], s=60, edgecolor=None)
        axes[1].set_aspect("equal")
        axes[1].set_facecolor(background_color)

        # Plot a circle around the network
        plot_network_contour(self.graph, axes[1], background_color=background_color)

        # Plot the labels, if any
        if labels:
            plot_labels(labels, self.graph, axes[1])

        if show_domain_ids:
            for domain in domains[domains > 0]:
                idx = self.node2domain["primary_domain"] == domain
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
                domain_color = np.reshape(domain2rgb[domain, :], (1, 4))

                alpha = node2nes.loc[:, domain].values
                alpha = alpha / self.enrichment_max_log10
                alpha[alpha > 1] = 1
                alpha = np.reshape(alpha, -1)

                c = np.repeat(domain_color, len(alpha), axis=0)
                # c[:, 3] = alpha

                idx = self.node2domain["primary_domain"] == domain
                # ix = np.argsort(c)
                axes[1 + domain].scatter(
                    node_xy[idx, 0], node_xy[idx, 1], c=c[idx], s=60, edgecolor=None
                )
                axes[1 + domain].set_aspect("equal")
                axes[1 + domain].set_facecolor(background_color)
                axes[1 + domain].set_title(
                    "Domain %d\n%s" % (domain, self.domains.loc[domain, "label"]),
                    color=foreground_color,
                )
                plot_network_contour(
                    self.graph, axes[1 + domain], background_color=background_color
                )

                # Plot the labels, if any
                if labels:
                    plot_labels(labels, self.graph, axes[1 + domain])

        fig.set_facecolor(background_color)

        if save_fig:
            path_to_fig = save_fig
            print("Output path: %s" % path_to_fig)
            plt.savefig(path_to_fig, facecolor=background_color)

    def plot_sample_attributes(
        self,
        attributes=1,
        top_attributes_only=False,
        show_network=True,
        show_costanzo2016=False,
        show_costanzo2016_colors=True,
        show_costanzo2016_clabels=False,
        show_nes=True,
        show_raw_data=False,
        show_significant_nodes=False,
        show_colorbar=True,
        colors=["82add6", "facb66"],
        background_color="#000000",
        labels=[],
        save_fig=None,
        **kwargs,
    ):
        foreground_color = "#ffffff"
        if background_color == "#ffffff":
            foreground_color = "#000000"

        all_attributes = self.attributes_matrix.index.values
        if top_attributes_only:
            all_attributes = all_attributes[self.attributes_matrix["top"]]

        if isinstance(attributes, int):
            if attributes < len(all_attributes):
                attributes = np.random.choice(all_attributes, attributes, replace=False)
            else:
                attributes = np.arange(len(all_attributes))
        elif isinstance(attributes, str):
            attributes = [list(self.attributes_matrix["name"].values).index(attributes)]
        elif isinstance(attributes, list):
            attributes = [
                list(self.attributes_matrix["name"].values).index(attribute)
                for attribute in attributes
            ]

        node_xy = get_node_coordinates(self.graph)

        # Figure parameters

        nax = 0
        if show_network:
            nax = 1

        nrows = int(np.ceil((len(attributes) + nax) / 2))
        ncols = np.min([len(attributes) + nax, 2])
        figsize = (10 * ncols, 10 * nrows)

        [fig, axes] = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=True,
            sharey=True,
            facecolor=background_color,
        )

        if isinstance(axes, np.ndarray):
            axes = axes.ravel()
        else:
            axes = np.array([axes])

        # First, plot the network (if required)
        if show_network:
            ax = axes[0]
            ax = plot_network(self.graph, ax=ax, background_color=background_color)

        score = self.nes

        # Plot the attribute
        for idx_attribute, attribute in enumerate(attributes):
            ax = axes[idx_attribute + nax]

            if show_nes:
                # Dynamically determine the min & max of the colorscale
                if "vmin" in kwargs:
                    vmin = kwargs["vmin"]
                else:
                    vmin = np.nanmin(
                        [
                            np.log10(1 / self.num_permutations),
                            np.nanmin(-np.abs(score[:, attribute])),
                        ]
                    )
                if "vmax" in kwargs:
                    vmax = kwargs["vmax"]
                else:
                    vmax = np.nanmax(
                        [
                            -np.log10(1 / self.num_permutations),
                            np.nanmax(np.abs(score[:, attribute])),
                        ]
                    )
                if "midrange" in kwargs:
                    midrange = kwargs["midrange"]
                else:
                    midrange = [np.log10(0.05), 0, -np.log10(0.05)]

                # Determine the order of points, such that the brightest ones are on top
                idx = np.argsort(np.abs(score[:, attribute]))

                # Colormap
                colors_hex = [
                    colors[0],
                    background_color,
                    background_color,
                    background_color,
                    colors[1],
                ]
                colors_hex = [re.sub(r"^#", "", c) for c in colors_hex]
                colors_rgb = [
                    tuple(int(c[i : i + 2], 16) / 255 for i in (0, 2, 4)) for c in colors_hex
                ]

                cmap = LinearSegmentedColormap.from_list("my_cmap", colors_rgb)

                sc = ax.scatter(
                    node_xy[idx, 0],
                    node_xy[idx, 1],
                    c=score[idx, attribute],
                    s=60,
                    cmap=cmap,
                    norm=MidpointRangeNormalize(midrange=midrange, vmin=vmin, vmax=vmax),
                    edgecolors=None,
                )

            if show_colorbar:
                pos_ax = ax.get_position()
                w = pos_ax.width * 0.75
                x0 = pos_ax.x0 + (pos_ax.width - w) / 2
                pos_cax = [x0, pos_ax.y0, w, pos_ax.height * 0.05]
                cax = fig.add_axes(pos_cax)

                cb = plt.colorbar(
                    sc,
                    cax=cax,
                    orientation="horizontal",
                    ticks=[vmin, midrange[0], midrange[1], midrange[2], vmax],
                    drawedges=False,
                )

                # pad = 0, shrink = 1,
                # set colorbar label plus label color
                cb.set_label("Neighborhood enrichment p-value", color=foreground_color)

                # set colorbar tick color
                cax.xaxis.set_tick_params(color=foreground_color)

                # set colorbar edgecolor
                cb.outline.set_edgecolor(foreground_color)
                cb.outline.set_linewidth(1)

                # set colorbar ticklabels
                plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color=foreground_color)

                cb.ax.set_xticklabels(
                    [
                        format(r"$10^{%d}$" % vmin),
                        r"$10^{%d}$" % midrange[0],
                        r"$1$",
                        r"$10^{%d}$" % -midrange[2],
                        format(r"$10^{-%d}$" % vmax),
                    ]
                )

                cax.text(
                    cax.get_xlim()[0],
                    1,
                    "Lower than random",
                    verticalalignment="bottom",
                    fontdict={"color": foreground_color},
                )
                cax.text(
                    cax.get_xlim()[1],
                    1,
                    "Higher than random",
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    fontdict={"color": foreground_color},
                )

            if show_raw_data:
                with np.errstate(divide="ignore", invalid="ignore"):
                    [s_zero, s_min, s_max] = [5, 5, 55]
                    n = self.node2attribute[:, attribute]

                    n2a = np.abs(n)
                    if set(np.unique(n2a[~np.isnan(n2a)])).issubset([0, 1]):
                        # The attribute is binary
                        s = np.zeros(len(n2a))
                        s[n2a > 0] = s_max
                        n_min = 0
                        n_max = 1
                    else:
                        # The attribute is quantitative
                        [n_min, n_max] = np.nanpercentile(np.unique(n2a), [5, 95])
                        a = (s_max - s_min) / (n_max - n_min)
                        b = s_min - a * n_min
                        s = a * n2a + b
                        s[s < s_min] = s_min
                        s[s > s_max] = s_max

                    # Colormap
                    [neg_color, pos_color, zero_color] = [
                        "#ff1d23",
                        "#00ff44",
                        foreground_color,
                    ]  # red, green, white

                    idx = self.node2attribute[:, attribute] < 0
                    sc1 = ax.scatter(
                        node_xy[idx, 0], node_xy[idx, 1], s=s[idx], c=neg_color, marker="."
                    )

                    idx = self.node2attribute[:, attribute] > 0
                    sc2 = ax.scatter(
                        node_xy[idx, 0], node_xy[idx, 1], s=s[idx], c=pos_color, marker="."
                    )

                    idx = self.node2attribute[:, attribute] == 0
                    sc3 = ax.scatter(
                        node_xy[idx, 0], node_xy[idx, 1], s=s_zero, c=zero_color, marker="."
                    )

                    # Legend
                    l1 = plt.scatter([], [], s=s_max, c=pos_color, edgecolors="none")
                    l2 = plt.scatter([], [], s=s_min, c=pos_color, edgecolors="none")
                    l3 = plt.scatter([], [], s=s_zero, c=zero_color, edgecolors="none")
                    l4 = plt.scatter([], [], s=s_min, c=neg_color, edgecolors="none")
                    l5 = plt.scatter([], [], s=s_max, c=neg_color, edgecolors="none")

                    legend_labels = ["{0:.2f}".format(n) for n in [n_max, n_min, 0, -n_min, -n_max]]

                    leg = ax.legend(
                        [l1, l2, l3, l4, l5],
                        legend_labels,
                        loc="upper left",
                        bbox_to_anchor=(0, 1),
                        title="Raw data",
                        scatterpoints=1,
                        fancybox=False,
                        facecolor=background_color,
                        edgecolor=background_color,
                    )

                    for leg_txt in leg.get_texts():
                        leg_txt.set_color(foreground_color)

                    leg_title = leg.get_title()
                    leg_title.set_color(foreground_color)

            if show_significant_nodes:
                with np.errstate(divide="ignore", invalid="ignore"):
                    idx = np.abs(self.nes_binary[:, attribute]) > 0
                    sn1 = ax.scatter(node_xy[idx, 0], node_xy[idx, 1], c="w", marker="+")

                # Legend
                s = "p < %.2e" % self.enrichment_threshold
                leg = ax.legend(
                    [sn1],
                    [s],
                    loc="upper left",
                    bbox_to_anchor=(0, 1),
                    title="Significance",
                    scatterpoints=1,
                    fancybox=False,
                    facecolor=background_color,
                    edgecolor=background_color,
                )

                for leg_txt in leg.get_texts():
                    leg_txt.set_color(foreground_color)

                leg_title = leg.get_title()
                leg_title.set_color(foreground_color)

            if show_costanzo2016:
                plot_costanzo2016_network_annotations(
                    self.graph,
                    ax,
                    self.path_to_safe_data,
                    colors=show_costanzo2016_colors,
                    clabels=show_costanzo2016_clabels,
                    background_color=background_color,
                )

            # Plot a circle around the network
            plot_network_contour(self.graph, ax, background_color=background_color)

            # Plot the labels, if any
            if labels:
                plot_labels(labels, self.graph, ax)

            ax.set_aspect("equal")
            ax.set_facecolor(background_color)

            ax.grid(False)
            ax.margins(0.1, 0.1)

            if idx_attribute + nax == 0:
                ax.invert_yaxis()

            title = self.attributes_matrix.loc[attribute, "name"]

            title = "\n".join(textwrap.wrap(title, width=30))
            ax.set_title(title, color=foreground_color)

            ax.set_frame_on(False)

        fig.set_facecolor(background_color)

        if save_fig:
            path_to_fig = save_fig
            if not os.path.isabs(path_to_fig):
                path_to_fig = os.path.join(self.output_dir, save_fig)
            print("Output path: %s" % path_to_fig)
            plt.savefig(path_to_fig, facecolor=background_color)

    def print_output_files(self, **kwargs):
        if "output_dir" in kwargs:
            self.output_dir = kwargs["output_dir"]

        # Domain properties
        path_domains = os.path.join(self.output_dir, "domain_properties_annotation.txt")
        if self.domains is not None:
            self.domains.drop(labels=[0], axis=0, inplace=True, errors="ignore")
            self.domains.to_csv(path_domains, sep="\t")
            print(path_domains)

        # Attribute properties
        path_attributes = os.path.join(self.output_dir, "attribute_properties_annotation.txt")
        self.attributes_matrix.to_csv(path_attributes, sep="\t")
        print(path_attributes)

        # Node properties
        path_nodes = os.path.join(self.output_dir, "node_properties_annotation.txt")

        t = nx.get_node_attributes(self.graph, "key")
        ids = list(t.keys())
        keys = list(t.values())
        t = nx.get_node_attributes(self.graph, "label")
        labels = list(t.values())
        if self.node2domain is not None:
            domains = self.node2domain["primary_domain"].values
            ness = self.node2domain["primary_nes"].values
            num_domains = self.node2domain[self.domains["id"]].sum(axis=1).values
            self.nodes = pd.DataFrame(
                data={
                    "id": ids,
                    "key": keys,
                    "label": labels,
                    "domain": domains,
                    "nes": ness,
                    "num_domains": num_domains,
                }
            )
        else:
            self.nodes = pd.DataFrame(self.nes)
            self.nodes.columns = self.attributes_matrix["name"]
            self.nodes.insert(loc=0, column="key", value=keys)
            self.nodes.insert(loc=1, column="label", value=labels)

        self.nodes.to_csv(path_nodes, sep="\t")
        print(path_nodes)


def run_safe_batch(attribute_file):
    sf = SAFE()
    sf.load_network()
    sf.define_neighborhoods()

    sf.load_network_annotation(attribute_file=attribute_file)
    sf.compute_pvalues(num_permutations=1000)

    return sf.nes


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Run Spatial Analysis of Functional Enrichment (SAFE) on the default Costanzo et al., 2016 network"
    )
    parser.add_argument(
        "path_to_attribute_file",
        metavar="path_to_attribute_file",
        type=str,
        help="Path to the file containing label-to-attribute annotation",
    )

    args = parser.parse_args()

    # Load the attribute file
    [attributes, node_label_order, node2attribute] = load_network_annotation(
        args.path_to_attribute_file
    )

    nr_processes = mp.cpu_count()
    nr_attributes = attributes.shape[0]

    chunk_size = np.ceil(nr_attributes / nr_processes).astype(int)
    chunks = np.array_split(np.arange(nr_attributes), nr_processes)

    all_chunks = []
    for chunk in chunks:
        this_chunk = pd.DataFrame(
            data=node2attribute[:, chunk],
            index=node_label_order,
            columns=attributes["name"].values[chunk],
        )
        all_chunks.append(this_chunk)

    pool = mp.Pool(processes=nr_processes)

    combined_nes = []

    print("Running SAFE on %d chunks of size %d..." % (nr_processes, chunk_size))
    for res in pool.map_async(run_safe_batch, all_chunks).get():
        combined_nes.append(res)

    all_nes = np.concatenate(combined_nes, axis=1)

    output_file = format("%s_safe_nes.p" % args.path_to_attribute_file)

    print("Saving the results...")
    with open(output_file, "wb") as handle:
        pickle.dump(all_nes, handle)
