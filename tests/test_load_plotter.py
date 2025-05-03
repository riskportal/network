"""
tests/test_load_plotter
~~~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import pytest

# NOTE: Displaying plots during testing can cause the program to hang. Avoid including plot displays in tests.
# Now, let's test the plotter with different custom settings.


def initialize_plotter(risk, graph):
    """
    Initialize the plotter with specified settings.

    Args:
        risk: The RISK object instance used for plotting.
        graph: The graph object to be plotted.

    Returns:
        Plotter: The initialized plotter object.
    """
    return risk.load_plotter(
        graph=graph,
        figsize=(15, 15),
        background_color="black",
        background_alpha=1.0,
    )


def plot_title(plotter):
    """
    Plot a title and subtitle on the network graph with preset parameters.

    Args:
        plotter: The initialized plotter object.
    """
    try:
        plotter.plot_title(
            title="Yeast Protein-Protein Interaction Network",
            subtitle="Michaelis et al., 2023",
            title_fontsize=20,
            subtitle_fontsize=14,
            font="Arial",
            title_color="black",
            subtitle_color="gray",
            title_x=0.5,
            title_y=0.975,
            title_space_offset=0.075,
            subtitle_offset=0.025,
        )
    finally:
        plt.close("all")


def plot_circle_perimeter(plotter):
    """
    Plot a circle perimeter around the network using the plotter.

    Args:
        plotter: The initialized plotter object.
    """
    try:
        plotter.plot_circle_perimeter(
            scale=1.05,
            center_offset_x=0.0,
            center_offset_y=0.0,
            linestyle="dashed",
            linewidth=1.5,
            color="black",
            outline_alpha=1.0,
            fill_alpha=0.0,
        )
    finally:
        plt.close("all")


def plot_contour_perimeter(plotter):
    """
    Plot a contour perimeter around the network using KDE-based contour.

    Args:
        plotter: The initialized plotter object.
    """
    try:
        plotter.plot_contour_perimeter(
            scale=1.0,
            levels=5,
            bandwidth=0.8,
            grid_size=250,
            color="black",
            linestyle="solid",
            linewidth=1.5,
            outline_alpha=1.0,
            fill_alpha=0.0,
        )
    finally:
        plt.close("all")


def plot_network(plotter):
    """
    Plot the full network using the plotter.

    Args:
        plotter: The initialized plotter object.
    """
    try:
        plotter.plot_network(
            node_size=plotter.get_annotated_node_sizes(
                significant_size=100, nonsignificant_size=25
            ),
            node_shape="o",
            edge_width=0.0,
            node_color=plotter.get_annotated_node_colors(
                cmap="gist_rainbow",
                blend_colors=True,
                blend_gamma=2.2,
                min_scale=0.25,
                max_scale=1.0,
                scale_factor=0.5,
                alpha=1.0,
                nonsignificant_color="white",
                nonsignificant_alpha=0.1,
                ids_to_colors=None,
                random_seed=887,
            ),
            node_edgecolor="black",
            edge_color="white",
            node_alpha=0.1,
            edge_alpha=1.0,
        )
    finally:
        plt.close("all")


def plot_subnetwork(plotter):
    """
    Plot a specific subnetwork using the plotter.

    Args:
        plotter: The initialized plotter object.
    """
    try:
        plotter.plot_subnetwork(
            nodes=[
                "LSM1",
                "LSM2",
                "LSM3",
                "LSM4",
                "LSM5",
                "LSM6",
                "LSM7",
                "PAT1",
            ],
            node_size=250,
            node_shape="^",
            edge_width=0.5,
            node_color="skyblue",
            node_edgecolor="black",
            edge_color="white",
            node_alpha=0.5,
            edge_alpha=0.5,
        )
    finally:
        plt.close("all")


def plot_contours(plotter):
    """
    Plot contours on the network using the plotter.

    Args:
        plotter: The initialized plotter object.
    """
    try:
        plotter.plot_contours(
            levels=5,
            bandwidth=0.8,
            grid_size=250,
            alpha=0.2,
            color=plotter.get_annotated_contour_colors(
                cmap="gist_rainbow", ids_to_colors=None, random_seed=887
            ),
        )
    finally:
        plt.close("all")


def plot_subcontour(plotter):
    """
    Plot subcontours on a specific subnetwork using the plotter.

    Args:
        plotter: The initialized plotter object.
    """
    try:
        plotter.plot_subcontour(
            nodes=[
                "LSM1",
                "LSM2",
                "LSM3",
                "LSM4",
                "LSM5",
                "LSM6",
                "LSM7",
                "PAT1",
            ],
            levels=5,
            bandwidth=0.8,
            grid_size=250,
            alpha=0.2,
            color="white",
        )
    finally:
        plt.close("all")


def plot_labels(plotter):
    """
    Plot labels on the network using the plotter.

    Args:
        plotter: The initialized plotter object.
    """
    try:
        plotter.plot_labels(
            scale=1.25,
            offset=0.10,
            font="Arial",
            fontcase={"lower": "title"},
            fontsize=10,
            fontcolor=plotter.get_annotated_label_colors(
                cmap="gist_rainbow", ids_to_colors=None, random_seed=887
            ),
            fontalpha=None,
            arrow_linewidth=1,
            arrow_style="->",
            arrow_color=plotter.get_annotated_label_colors(
                cmap="gist_rainbow", ids_to_colors=None, random_seed=887
            ),
            arrow_alpha=None,
            arrow_base_shrink=0.0,
            arrow_tip_shrink=0.0,
            max_labels=10,
            min_label_lines=2,
            max_label_lines=4,
            min_chars_per_line=3,
            max_chars_per_line=20,
            words_to_omit=["process", "biosynthetic"],
            overlay_ids=False,
            ids_to_keep=None,
            ids_to_labels=None,
        )
    finally:
        plt.close("all")


def plot_sublabel(plotter):
    """
    Plot a specific sublabel on the network using the plotter.

    Args:
        plotter: The initialized plotter object.
    """
    try:
        plotter.plot_sublabel(
            nodes=[
                "LSM1",
                "LSM2",
                "LSM3",
                "LSM4",
                "LSM5",
                "LSM6",
                "LSM7",
                "PAT1",
            ],
            label="LSM1-7-PAT1 Complex",
            radial_position=73,
            scale=1.6,
            offset=0.10,
            font="Arial",
            fontsize=14,
            fontcolor="white",
            arrow_linewidth=1.5,
            arrow_color="white",
            fontalpha=0.5,
            arrow_alpha=0.5,
        )
    finally:
        plt.close("all")


def test_initialize_plotter(risk_obj, graph):
    """
    Test initializing the plotter object with a graph.

    Args:
        risk_obj: The RISK object instance used for initializing the plotter.
        graph: The graph object to be plotted.
    """
    plotter = initialize_plotter(risk_obj, graph)

    assert plotter is not None  # Ensure the plotter is initialized
    assert hasattr(plotter, "graph")  # Check that the plotter has a graph attribute


def test_plot_title(risk_obj, graph):
    """
    Test the basic plotting of title and subtitle on the network graph.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_title(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_circle_perimeter(risk_obj, graph):
    """
    Test plotting a circle perimeter around the network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_circle_perimeter(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_contour_perimeter(risk_obj, graph):
    """
    Test plotting a contour perimeter around the network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_contour_perimeter(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_network(risk_obj, graph):
    """
    Test plotting the full network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_network(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_subnetwork(risk_obj, graph):
    """
    Test plotting a subnetwork using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object containing the subnetwork to be plotted.
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_subnetwork(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_contours(risk_obj, graph):
    """
    Test plotting contours on the network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which contours will be plotted.
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_contours(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_subcontour(risk_obj, graph):
    """
    Test plotting subcontours on the network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which subcontours will be plotted.
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_subcontour(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_labels(risk_obj, graph):
    """
    Test plotting labels on the network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which labels will be plotted.
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_labels(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_sublabel(risk_obj, graph):
    """
    Test plotting a sublabel on the network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which the sublabel will be plotted.
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_sublabel(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


@pytest.mark.parametrize(
    "title, subtitle, title_fontsize, subtitle_fontsize, title_color, subtitle_color, title_x, title_y, title_space_offset, subtitle_offset, font",
    [
        (
            "Metabolic Network",
            "Enrichment Analysis",
            16,
            12,
            "white",
            "red",
            0.5,
            0.95,
            0.05,
            0.03,
            "Arial",
        ),  # Test case 1
        (
            "Cluster Analysis",
            "K-means Results",
            20,
            14,
            "yellow",
            "blue",
            0.3,
            0.975,
            0.075,
            0.025,
            "Verdana",
        ),  # Test case 2
    ],
)
def test_plot_title_with_custom_params(
    risk_obj,
    graph,
    title,
    subtitle,
    title_fontsize,
    subtitle_fontsize,
    title_color,
    subtitle_color,
    title_x,
    title_y,
    title_space_offset,
    subtitle_offset,
    font,
):
    """
    Test the plot_title method with different title and subtitle configurations.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
        title (str): The title of the plot.
        subtitle (str): The subtitle of the plot.
        title_fontsize (int): Font size of the title.
        subtitle_fontsize (int): Font size of the subtitle.
        title_color (str): Color of the title text.
        subtitle_color (str): Color of the subtitle text.
        title_x (float): Position of the title in figure coordinates (0-1).
        title_y (float): Position of the title in figure coordinates (0-1).
        title_space_offset (float): Fraction of figure height to leave for the space above the plot.
        subtitle_offset (float): Offset factor to position the subtitle below the title.
        font (str): Font family used for both title and subtitle.
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        plotter.plot_title(
            title=title,
            subtitle=subtitle,
            title_fontsize=title_fontsize,
            subtitle_fontsize=subtitle_fontsize,
            title_color=title_color,
            subtitle_color=subtitle_color,
            title_x=title_x,
            title_y=title_y,
            title_space_offset=title_space_offset,
            subtitle_offset=subtitle_offset,
            font=font,
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "color, outline_alpha, fill_alpha, scale, linestyle, linewidth, center_offset_x, center_offset_y",
    [
        (
            "white",
            None,
            None,
            1.05,
            "solid",
            1.5,
            0.0,
            0.0,
        ),  # Test case 1
        (
            (0.5, 0.8, 1.0),
            0.5,
            0.5,
            1.1,
            "dashed",
            2.0,
            -0.2,
            0.1,
        ),  # Test case 2
        (
            "black",
            0.8,
            0.3,
            0.9,
            "dotted",
            1.0,
            0.5,
            -0.3,
        ),  # Test case 3
    ],
)
def test_plot_circle_perimeter_with_custom_params(
    risk_obj,
    graph,
    color,
    outline_alpha,
    fill_alpha,
    scale,
    linestyle,
    linewidth,
    center_offset_x,
    center_offset_y,
):
    """
    Test plot_circle_perimeter with different color, alpha, and style parameters.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
        color: The color parameter for the perimeter.
        outline_alpha: The transparency of the perimeter (outline).
        fill_alpha: The transparency of the circle's fill.
        scale: Scaling factor for the perimeter diameter.
        linestyle: The line style for the circle's outline.
        linewidth: The thickness of the circle's outline.
        center_offset_x: The x-coordinate offset of the circle's center.
        center_offset_y: The y-coordinate offset of the circle's center.
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        plotter.plot_circle_perimeter(
            scale=scale,
            center_offset_x=center_offset_x,
            center_offset_y=center_offset_y,
            linestyle=linestyle,
            linewidth=linewidth,
            color=color,
            outline_alpha=outline_alpha,
            fill_alpha=fill_alpha,
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "scale, levels, bandwidth, grid_size, color, linestyle, linewidth, outline_alpha, fill_alpha",
    [
        (
            1.0,
            3,
            0.8,
            250,
            "black",
            "solid",
            1.5,
            None,
            None,
        ),  # Test case 1
        (
            1.1,
            5,
            1.0,
            300,
            (0.5, 0.8, 1.0),
            "dashed",
            2.0,
            1.0,
            0.5,
        ),  # Test case 2
    ],
)
def test_plot_contour_perimeter_with_custom_params(
    risk_obj,
    graph,
    scale,
    levels,
    bandwidth,
    grid_size,
    color,
    linestyle,
    linewidth,
    outline_alpha,
    fill_alpha,
):
    """
    Test plot_contour_perimeter with different contour and alpha parameters.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
        scale: Scaling factor for the contour size.
        levels: Number of contour levels.
        bandwidth: Bandwidth for the KDE smoothing.
        grid_size: Grid size for the KDE computation.
        color: Color of the contour perimeter.
        linestyle: Line style of the contour.
        linewidth: Line width of the contour perimeter.
        outline_alpha: Transparency of the contour outline.
        fill_alpha: Transparency of the contour fill.
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        plotter.plot_contour_perimeter(
            scale=scale,
            levels=levels,
            bandwidth=bandwidth,
            grid_size=grid_size,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            outline_alpha=outline_alpha,
            fill_alpha=fill_alpha,
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "node_color, cmap, nonsignificant_color, nonsignificant_alpha, ids_to_colors, edge_color, node_edgecolor, node_alpha, edge_alpha, node_size, node_shape, edge_width, node_edgewidth",
    [
        (
            None,
            "gist_rainbow",
            "white",
            None,
            None,
            "black",
            "blue",
            None,
            None,
            100,
            "o",
            0.0,
            1.5,
        ),
        (
            [(0.2, 0.6, 0.8)],
            None,
            (1.0, 1.0, 0.5),
            0.5,
            {1: "orange"},
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            0.5,
            0.5,
            150,
            "^",
            1.0,
            2.0,
        ),
        (
            ["green", "blue", "red"],
            None,
            "yellow",
            0.2,
            {1: (1.0, 0.0, 0.0)},
            "grey",
            "red",
            0.8,
            0.9,
            120,
            "s",
            2.0,
            3.0,
        ),
        (
            [(0.1, 0.2, 0.3, 0.8), (0.3, 0.4, 0.5, 1.0)],
            None,
            "yellow",
            0.4,
            {1: (0.1, 0.2, 0.3, 0.8)},
            "grey",
            "black",
            0.7,
            0.5,
            100,
            "o",
            1.0,
            2.0,
        ),
        (
            ["red", "green", "blue"],
            "viridis",
            "white",
            None,
            {1: "red"},
            "black",
            "blue",
            None,
            None,
            100,
            "o",
            1.0,
            2.0,
        ),
    ],
)
def test_plot_network_with_custom_params(
    risk_obj,
    graph,
    node_color,
    cmap,
    nonsignificant_color,
    nonsignificant_alpha,
    ids_to_colors,
    edge_color,
    node_edgecolor,
    node_alpha,
    edge_alpha,
    node_size,
    node_shape,
    edge_width,
    node_edgewidth,
):
    """
    Test plot_network with different node, edge, and node edge colors, sizes, edge widths, node edge widths, and alpha values.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
        node_color: The color of the network nodes.
        cmap: Colormap to use for node colors if node_color is None.
        nonsignificant_color: The color for non-significant nodes.
        nonsignificant_alpha: The transparency of the non-significant nodes.
        ids_to_colors: A dictionary mapping domain IDs to specific colors.
        edge_color: The color of the network edges.
        node_edgecolor: The color of the node edges.
        node_alpha: The transparency of the nodes.
        edge_alpha: The transparency of the edges.
        node_size: The size of the nodes.
        node_shape: The shape of the nodes.
        edge_width: The width of the edges.
        node_edgewidth: The width of the edges around the nodes.
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        # Test different color formats
        node_colors = [
            plotter.get_annotated_node_colors(
                cmap=cmap,
                color=node_color,
                blend_colors=True,
                blend_gamma=2.2,
                min_scale=0.5,
                max_scale=1.0,
                scale_factor=0.5,
                nonsignificant_color=nonsignificant_color,
                nonsignificant_alpha=nonsignificant_alpha,
                ids_to_colors=ids_to_colors,
                random_seed=887,
            ),
            node_color,
        ]
        for node_color in node_colors:
            plotter.plot_network(
                node_size=plotter.get_annotated_node_sizes(
                    significant_size=100, nonsignificant_size=node_size
                ),
                edge_width=edge_width,
                node_color=node_color,
                node_edgecolor=node_edgecolor,
                node_edgewidth=node_edgewidth,
                edge_color=edge_color,
                node_shape=node_shape,
                node_alpha=node_alpha,
                edge_alpha=edge_alpha,
            )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "node_color, edge_color, node_edgecolor, node_size, edge_width, node_alpha, edge_alpha, node_shape",
    [
        ("white", "black", "blue", 250, 0.0, None, None, "^"),
        ((0.2, 0.6, 0.8), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 300, 0.5, 0.5, 0.5, "s"),
        ("green", "gray", "red", 200, 1.0, 0.8, 0.9, "o"),
    ],
)
def test_plot_subnetwork_with_custom_params(
    risk_obj,
    graph,
    node_color,
    edge_color,
    node_edgecolor,
    node_size,
    edge_width,
    node_alpha,
    edge_alpha,
    node_shape,
):
    """
    Test plot_subnetwork with different node, edge, and node edge colors, sizes, edge widths, shapes, and alpha values.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
        node_color: The color of the network nodes.
        edge_color: The color of the network edges.
        node_edgecolor: The color of the node edges.
        node_size: The size of the nodes.
        edge_width: The width of the edges.
        node_alpha: The transparency of the nodes.
        edge_alpha: The transparency of the edges.
        node_shape: The shape of the nodes.
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        # Nodes are grouped into a single list
        plotter.plot_subnetwork(
            nodes=[
                "LSM1",
                "LSM2",
                "LSM3",
                "LSM4",
                "LSM5",
                "LSM6",
                "LSM7",
                "PAT1",
            ],
            node_size=node_size,
            edge_width=edge_width,
            node_color=node_color,
            node_edgecolor=node_edgecolor,
            edge_color=edge_color,
            node_shape=node_shape,
            node_alpha=node_alpha,
            edge_alpha=edge_alpha,
        )
        # Nodes are grouped into two lists
        plotter.plot_subnetwork(
            nodes=[
                [
                    "LSM1",
                    "LSM2",
                    "LSM3",
                ],
                [
                    "LSM4",
                    "LSM5",
                    "LSM6",
                    "LSM7",
                    "PAT1",
                ],
            ],
            node_size=node_size,
            edge_width=edge_width,
            node_color=node_color,
            node_edgecolor=node_edgecolor,
            edge_color=edge_color,
            node_shape=node_shape,
            node_alpha=node_alpha,
            edge_alpha=edge_alpha,
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "color, alpha, fill_alpha, levels, bandwidth, grid_size, linestyle, linewidth",
    [
        (None, None, None, 5, 0.8, 250, "solid", 1.5),  # Test case with annotated colors
        ("red", 0.5, 0.3, 6, 1.0, 300, "dashed", 2.0),  # Single string color
        ((0.2, 0.6, 0.8), 0.3, 0.15, 4, 0.6, 200, "dotted", 1.0),  # RGB tuple
        (["red", "green", "blue"], 0.7, 0.4, 8, 1.2, 350, "dashdot", 1.8),  # List of string colors
        (
            [(0.1, 0.2, 0.3, 0.8), (0.4, 0.5, 0.6, 1.0)],
            0.9,
            0.5,
            10,
            0.9,
            280,
            "solid",
            1.2,
        ),  # List of RGBA colors
    ],
)
def test_plot_contours_with_custom_params(
    risk_obj,
    graph,
    color,
    alpha,
    fill_alpha,
    levels,
    bandwidth,
    grid_size,
    linestyle,
    linewidth,
):
    """
    Test plot_contours with different color formats, alpha values, fill_alpha values, contour levels, bandwidths, grid sizes, and line styles.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which contours will be plotted.
        color: The color of the contours (None for annotated, string, or RGB tuple).
        alpha: The transparency of the contour lines.
        fill_alpha: The transparency of the contour fill.
        levels: The number of contour levels.
        bandwidth: The bandwidth for KDE smoothing.
        grid_size: The grid size for contour resolution.
        linestyle: The line style for the contour lines.
        linewidth: The width of the contour lines.
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        # Test different color formats
        contour_colors = [
            plotter.get_annotated_contour_colors(
                cmap="gist_rainbow",
                color=color,
                blend_colors=False,
                blend_gamma=2.2,
                min_scale=0.8,
                max_scale=1.0,
                scale_factor=1.0,
                ids_to_colors=None,
                random_seed=887,
            ),
            color,
        ]
        for color in contour_colors:
            plotter.plot_contours(
                levels=levels,
                bandwidth=bandwidth,
                grid_size=grid_size,
                alpha=alpha,
                fill_alpha=fill_alpha,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "color, alpha, fill_alpha, levels, bandwidth, grid_size, linestyle, linewidth",
    [
        (
            "red",
            None,
            None,
            5,
            0.8,
            250,
            "solid",
            1.5,
        ),  # Test case 1
        (
            (0.2, 0.6, 0.8),
            0.3,
            0.1,
            4,
            1.0,
            300,
            "dashed",
            2.0,
        ),  # Test case 2
    ],
)
def test_plot_subcontour_with_custom_params(
    risk_obj,
    graph,
    color,
    alpha,
    fill_alpha,
    levels,
    bandwidth,
    grid_size,
    linestyle,
    linewidth,
):
    """
    Test plot_subcontour with different color formats, alpha values, fill_alpha values, contour levels, bandwidths, grid sizes, and line styles.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which subcontours will be plotted.
        color: The color of the subcontours (string or RGB tuple).
        alpha: The transparency of the contour lines.
        fill_alpha: The transparency of the contour fill.
        levels: The number of contour levels.
        bandwidth: The bandwidth for KDE smoothing.
        grid_size: The grid size for contour resolution.
        linestyle: The line style for the contour lines.
        linewidth: The width of the contour lines.
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        # Nodes are grouped into a single list
        plotter.plot_subcontour(
            nodes=[
                "LSM1",
                "LSM2",
                "LSM3",
                "LSM4",
                "LSM5",
                "LSM6",
                "LSM7",
                "PAT1",
            ],
            levels=levels,
            bandwidth=bandwidth,
            grid_size=grid_size,
            alpha=alpha,
            fill_alpha=fill_alpha,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        # Nodes are grouped into two lists
        plotter.plot_subcontour(
            nodes=[
                [
                    "LSM1",
                    "LSM2",
                    "LSM3",
                ],
                [
                    "LSM4",
                    "LSM5",
                    "LSM6",
                    "LSM7",
                    "PAT1",
                ],
            ],
            levels=levels,
            bandwidth=bandwidth,
            grid_size=grid_size,
            alpha=alpha,
            fill_alpha=fill_alpha,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "scale, offset, font, fontcase, fontsize, fontcolor, font_alpha, arrow_linewidth, arrow_style, arrow_color, arrow_alpha, arrow_base_shrink, arrow_tip_shrink, max_labels, min_label_lines, max_label_lines, min_chars_per_line, max_chars_per_line, overlay_ids, ids_to_keep, ids_to_labels",
    [
        (
            1.25,
            0.10,
            "Arial",
            None,
            10,
            None,
            None,
            1,
            "->",
            None,
            None,
            5,
            3,
            10,
            2,
            4,
            3,
            10,
            False,
            None,
            None,
        ),  # Test case 1 (annotated colors)
        (
            1.5,
            0.15,
            "Helvetica",
            None,
            12,
            "red",
            0.5,
            2,
            "-|>",
            "blue",
            0.5,
            8,
            4,
            5,
            1,
            5,
            4,
            15,
            True,
            ["LSM1", "LSM2"],
            None,
        ),  # Test case 2 (single color)
        (
            1.35,
            0.12,
            "Times New Roman",
            None,
            14,
            (0.2, 0.6, 0.8),
            0.3,
            1.5,
            "<|-",
            (1.0, 0.0, 0.0),
            0.3,
            10,
            5,
            8,
            3,
            6,
            2,
            20,
            False,
            ["LSM1", "LSM3"],
            {1: "custom label"},
        ),  # Test case 3 (RGB color)
        (
            1.4,
            0.11,
            "Courier New",
            None,
            11,
            ["red", "green", "blue"],
            0.7,
            1,
            "->",
            ["yellow", "cyan", "magenta"],
            0.7,
            7,
            4,
            6,
            2,
            5,
            3,
            12,
            True,
            None,
            None,
        ),  # Test case 4 (list of colors)
        (
            1.2,
            0.09,
            "Verdana",
            None,
            13,
            [(0.1, 0.2, 0.3, 0.8), (0.4, 0.5, 0.6, 1.0)],
            0.9,
            2,
            "-|>",
            [(0.5, 0.5, 0.5, 0.5), (0.8, 0.2, 0.3, 1.0)],
            0.9,
            9,
            5,
            7,
            3,
            6,
            4,
            18,
            False,
            None,
            None,
        ),  # Test case 5 (list of RGBA colors)
    ],
)
def test_plot_labels_with_custom_params(
    risk_obj,
    graph,
    scale,
    offset,
    font,
    fontcase,
    fontsize,
    fontcolor,
    font_alpha,
    arrow_linewidth,
    arrow_style,
    arrow_color,
    arrow_alpha,
    arrow_base_shrink,
    arrow_tip_shrink,
    max_labels,
    min_label_lines,
    max_label_lines,
    min_chars_per_line,
    max_chars_per_line,
    overlay_ids,
    ids_to_keep,
    ids_to_labels,
):
    """
    Test plot_labels with varying label and arrow customization options, including font colors, alpha values, label placement
        constraints, and additional style parameters.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which labels will be plotted.
        scale: Scale factor for positioning labels around the perimeter.
        offset: Offset distance for labels from the perimeter.
        font: Font name for the labels.
        fontcase: Text transformation for the font (e.g., uppercase, lowercase).
        fontsize: Font size for the labels.
        fontcolor: The color of the label text (None for annotated, string, or RGB tuple).
        font_alpha: The transparency of the label text.
        arrow_linewidth: Line width for the arrows pointing to centroids.
        arrow_style: The style of the arrows (e.g., '->', '-|>', '<|-').
        arrow_color: The color of the label arrows (None for annotated, string, or RGB tuple).
        arrow_alpha: The transparency of the label arrows.
        arrow_base_shrink: Distance between the text and the base of the arrow.
        arrow_tip_shrink: Distance between the tip of the arrow and the centroid.
        max_labels: Maximum number of labels to display.
        min_label_lines: Minimum number of words per label.
        max_label_lines: Maximum number of words per label.
        min_chars_per_line: Minimum character count per word in the label.
        max_chars_per_line: Maximum character count per word in the label.
        overlay_ids: Whether to overlay domain IDs in the center of the centroids.
        ids_to_keep: List of IDs to prioritize for labeling.
        ids_to_labels: Dictionary mapping domain IDs to custom labels.
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        # Test different color formats
        label_colors = [
            plotter.get_annotated_label_colors(
                cmap="gist_rainbow",
                color=fontcolor,
                blend_colors=True,
                blend_gamma=2.2,
                min_scale=0.5,
                max_scale=1.0,
                scale_factor=0.5,
                ids_to_colors=None,
                random_seed=887,
            ),
            fontcolor,
        ]
        arrow_colors = [
            plotter.get_annotated_label_colors(
                cmap="gist_rainbow",
                color=arrow_color,
                blend_colors=False,
                blend_gamma=2.2,
                min_scale=0.5,
                max_scale=1.0,
                scale_factor=0.5,
                ids_to_colors=None,
                random_seed=887,
            ),
            arrow_color,
        ]
        # Test different color formats
        for fontcolor, arrow_color in zip(label_colors, arrow_colors):
            plotter.plot_labels(
                scale=scale,
                offset=offset,
                font=font,
                fontcase=fontcase,
                fontsize=fontsize,
                fontcolor=fontcolor,
                fontalpha=font_alpha,
                arrow_linewidth=arrow_linewidth,
                arrow_style=arrow_style,
                arrow_color=arrow_color,
                arrow_alpha=arrow_alpha,
                arrow_base_shrink=arrow_base_shrink,
                arrow_tip_shrink=arrow_tip_shrink,
                max_labels=max_labels,
                max_label_lines=max_label_lines,
                min_label_lines=min_label_lines,
                max_chars_per_line=max_chars_per_line,
                min_chars_per_line=min_chars_per_line,
                words_to_omit=["process", "biosynthetic"],
                overlay_ids=overlay_ids,
                ids_to_keep=ids_to_keep,
                ids_to_labels=ids_to_labels,
            )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "fontcolor, arrow_color, font_alpha, arrow_alpha, fontsize, radial_position, scale, offset, arrow_linewidth, font, arrow_style, arrow_base_shrink, arrow_tip_shrink",
    [
        (
            "white",
            "white",
            None,
            None,
            14,
            73,
            1.6,
            0.10,
            1.5,
            "Arial",
            "->",
            10,
            5,
        ),  # Test case 1
        (
            "red",
            "blue",
            0.5,
            0.5,
            16,
            120,
            1.5,
            0.12,
            2.0,
            "Helvetica",
            "-[",
            15,
            10,
        ),  # Test case 2
        (
            (0.2, 0.6, 0.8),
            (1.0, 0.0, 0.0),
            0.8,
            0.7,
            18,
            45,
            1.4,
            0.08,
            1.8,
            "Times New Roman",
            "<|-",
            12,
            8,
        ),  # Test case 3
    ],
)
def test_plot_sublabel_with_custom_params(
    risk_obj,
    graph,
    fontcolor,
    arrow_color,
    font_alpha,
    arrow_alpha,
    fontsize,
    radial_position,
    scale,
    offset,
    arrow_linewidth,
    font,
    arrow_style,
    arrow_base_shrink,
    arrow_tip_shrink,
):
    """
    Test plot_sublabel with different label and arrow colors, alpha values, label positioning, and other style parameters.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which the sublabel will be plotted.
        fontcolor: The color of the label text (string or RGB tuple).
        arrow_color: The color of the label arrows (string or RGB tuple).
        font_alpha: The transparency of the label text.
        arrow_alpha: The transparency of the label arrows.
        fontsize: The size of the label text.
        radial_position: The radial position of the label.
        scale: Scale factor for the label's radial position.
        offset: Offset distance for labels from the perimeter.
        arrow_linewidth: The width of the arrow pointing to the label.
        font: The font used for the label text.
        arrow_style: The style of the arrow pointing to the label.
        arrow_base_shrink: Distance between the text and the base of the arrow.
        arrow_tip_shrink: Distance between the tip of the arrow and the centroid.
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        # Nodes are grouped into a single list
        plotter.plot_sublabel(
            nodes=[
                "LSM1",
                "LSM2",
                "LSM3",
                "LSM4",
                "LSM5",
                "LSM6",
                "LSM7",
                "PAT1",
            ],
            label="LSM1-7-PAT1 Complex",
            radial_position=radial_position,
            scale=scale,
            offset=offset,
            font=font,
            fontsize=fontsize,
            fontcolor=fontcolor,
            fontalpha=font_alpha,
            arrow_linewidth=arrow_linewidth,
            arrow_color=arrow_color,
            arrow_alpha=arrow_alpha,
            arrow_style=arrow_style,
            arrow_base_shrink=arrow_base_shrink,
            arrow_tip_shrink=arrow_tip_shrink,
        )
        # Nodes are grouped into two lists
        plotter.plot_sublabel(
            nodes=[
                [
                    "LSM1",
                    "LSM2",
                    "LSM3",
                ],
                [
                    "LSM4",
                    "LSM5",
                    "LSM6",
                    "LSM7",
                    "PAT1",
                ],
            ],
            label="LSM1-7-PAT1 Complex",
            radial_position=radial_position,
            scale=scale,
            offset=offset,
            font=font,
            fontsize=fontsize,
            fontcolor=fontcolor,
            fontalpha=font_alpha,
            arrow_linewidth=arrow_linewidth,
            arrow_color=arrow_color,
            arrow_alpha=arrow_alpha,
            arrow_style=arrow_style,
            arrow_base_shrink=arrow_base_shrink,
            arrow_tip_shrink=arrow_tip_shrink,
        )
    finally:
        plt.close("all")


def test_plotter_savefig(risk_obj, graph, tmp_path):
    """
    Test that savefig creates a valid output image file.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
        tmp_path: Temporary directory for saving the output image.
    """
    plotter = initialize_plotter(risk_obj, graph)
    output_path = tmp_path / "test_plot_output.png"
    try:
        plot_network(plotter)  # must create a figure first
        plotter.savefig(str(output_path), dpi=150)
        assert output_path.exists()
        assert output_path.stat().st_size > 0  # file isn't empty
    finally:
        plt.close("all")


def test_plotter_show(risk_obj, graph, monkeypatch):
    """
    Test that show() executes without raising exceptions.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
        monkeypatch: Pytest fixture to patch the plt.show method.
    """
    plotter = initialize_plotter(risk_obj, graph)
    # Patch plt.show to prevent GUI rendering
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        plot_network(plotter)
        plotter.show()  # Just call it — no assertion needed
    finally:
        plt.close("all")
