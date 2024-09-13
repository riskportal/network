"""
tests/test_load_plotter
~~~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
import pytest


def test_initialize_plotter(risk_obj, graph):
    """Test initializing the plotter object with a graph.

    Args:
        risk_obj: The RISK object instance used for initializing the plotter.
        graph: The graph object to be plotted.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)

    assert plotter is not None  # Ensure the plotter is initialized
    assert hasattr(plotter, "graph")  # Check that the plotter has a graph attribute


def test_plot_circle_perimeter(risk_obj, graph):
    """Test plotting a circle perimeter around the network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_circle_perimeter(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_contour_perimeter(risk_obj, graph):
    """Test plotting a contour perimeter around the network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_contour_perimeter(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_network(risk_obj, graph):
    """Test plotting the full network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_network(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_subnetwork(risk_obj, graph):
    """Test plotting a subnetwork using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object containing the subnetwork to be plotted.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_subnetwork(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_contours(risk_obj, graph):
    """Test plotting contours on the network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which contours will be plotted.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_contours(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_subcontour(risk_obj, graph):
    """Test plotting subcontours on the network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which subcontours will be plotted.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_subcontour(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_labels(risk_obj, graph):
    """Test plotting labels on the network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which labels will be plotted.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_labels(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def test_plot_sublabel(risk_obj, graph):
    """Test plotting a sublabel on the network using the plotter.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which the sublabel will be plotted.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    plot_sublabel(plotter)

    assert plotter is not None  # Ensure the plotter is initialized


def initialize_plotter(risk, graph):
    """Initialize the plotter with specified settings.

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
    )


def plot_circle_perimeter(plotter):
    """Plot a circle perimeter around the network using the plotter.

    Args:
        plotter: The initialized plotter object.

    Returns:
        None
    """
    try:
        plotter.plot_circle_perimeter(
            scale=1.05,
            linestyle="dashed",
            linewidth=1.5,
            color="black",
            outline_alpha=1.0,
            fill_alpha=0.0,
        )
    finally:
        plt.close("all")


def plot_contour_perimeter(plotter):
    """Plot a contour perimeter around the network using KDE-based contour.

    Args:
        plotter: The initialized plotter object.

    Returns:
        None
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
    """Plot the full network using the plotter.

    Args:
        plotter: The initialized plotter object.

    Returns:
        None
    """
    try:
        plotter.plot_network(
            node_size=plotter.get_annotated_node_sizes(enriched_size=100, nonenriched_size=25),
            node_shape="o",
            edge_width=0.0,
            node_color=plotter.get_annotated_node_colors(
                cmap="gist_rainbow",
                min_scale=0.25,
                max_scale=1.0,
                scale_factor=0.5,
                alpha=1.0,
                nonenriched_color="white",
                nonenriched_alpha=0.1,
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
    """Plot a specific subnetwork using the plotter.

    Args:
        plotter: The initialized plotter object.

    Returns:
        None
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
    """Plot contours on the network using the plotter.

    Args:
        plotter: The initialized plotter object.

    Returns:
        None
    """
    try:
        plotter.plot_contours(
            levels=5,
            bandwidth=0.8,
            grid_size=250,
            alpha=0.2,
            color=plotter.get_annotated_contour_colors(cmap="gist_rainbow", random_seed=887),
        )
    finally:
        plt.close("all")


def plot_subcontour(plotter):
    """Plot subcontours on a specific subnetwork using the plotter.

    Args:
        plotter: The initialized plotter object.

    Returns:
        None
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
    """Plot labels on the network using the plotter.

    Args:
        plotter: The initialized plotter object.

    Returns:
        None
    """
    try:
        plotter.plot_labels(
            scale=1.25,
            offset=0.10,
            font="Arial",
            fontsize=10,
            fontcolor=plotter.get_annotated_label_colors(cmap="gist_rainbow", random_seed=887),
            arrow_linewidth=1,
            arrow_color=plotter.get_annotated_label_colors(cmap="gist_rainbow", random_seed=887),
            max_labels=10,
            max_words=4,
            min_words=2,
            max_word_length=20,
            min_word_length=3,
            words_to_omit=["process", "biosynthetic"],
        )
    finally:
        plt.close("all")


def plot_sublabel(plotter):
    """Plot a specific sublabel on the network using the plotter.

    Args:
        plotter: The initialized plotter object.

    Returns:
        None
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


# NOTE: Displaying plots during testing can cause the program to hang. Avoid including plot displays in tests.
# Now, let's test the plotter with different custom settings.


@pytest.mark.parametrize(
    "color, outline_alpha, fill_alpha, scale, linestyle, linewidth",
    [
        (
            "white",
            1.0,
            0.0,
            1.05,
            "solid",
            1.5,
        ),  # Test case 1: White color, full perimeter opacity, no fill
        (
            (0.5, 0.8, 1.0),
            0.5,
            0.5,
            1.1,
            "dashed",
            2.0,
        ),  # Test case 2: Light blue RGB, semi-transparent perimeter and fill
    ],
)
def test_plot_circle_perimeter_with_custom_params(
    risk_obj, graph, color, outline_alpha, fill_alpha, scale, linestyle, linewidth
):
    """Test plot_circle_perimeter with different color, alpha, and style parameters.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
        color: The color parameter for the perimeter.
        outline_alpha: The transparency of the perimeter (outline).
        fill_alpha: The transparency of the circle's fill.
        scale: Scaling factor for the perimeter diameter.
        linestyle: The line style for the circle's outline.
        linewidth: The thickness of the circle's outline.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        plotter.plot_circle_perimeter(
            scale=scale,
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
            1.0,
            0.0,
        ),  # Test case 1: Black solid line, full opacity, no fill
        (
            1.1,
            5,
            1.0,
            300,
            (0.5, 0.8, 1.0),
            "dashed",
            2.0,
            0.7,
            0.5,
        ),  # Test case 2: Light blue contour with semi-transparent perimeter and fill
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
    """Test plot_contour_perimeter with different contour and alpha parameters.

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

    Returns:
        None
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
    "node_color, nonenriched_color, nonenriched_alpha, edge_color, node_edgecolor, node_alpha, edge_alpha, node_size, node_shape, edge_width, node_edgewidth",
    [
        (None, "white", 0.1, "black", "blue", 1.0, 1.0, 100, "o", 0.0, 1.5),  # Test case 1
        (
            (0.2, 0.6, 0.8),
            (1.0, 1.0, 0.5),
            0.5,
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            0.5,
            0.5,
            150,
            "^",
            1.0,
            2.0,
        ),  # Test case 2
        ("green", "yellow", 0.2, "grey", "red", 0.8, 0.9, 120, "s", 2.0, 3.0),  # Test case 3
    ],
)
def test_plot_network_with_custom_params(
    risk_obj,
    graph,
    node_color,
    nonenriched_color,
    nonenriched_alpha,
    edge_color,
    node_edgecolor,
    node_alpha,
    edge_alpha,
    node_size,
    node_shape,
    edge_width,
    node_edgewidth,
):
    """Test plot_network with different node, edge, and node edge colors, sizes, edge widths, node edge widths, and alpha values.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
        node_color: The color of the network nodes.
        nonenriched_color: The color for non-enriched nodes.
        nonenriched_alpha: The transparency of the non-enriched nodes.
        edge_color: The color of the network edges.
        node_edgecolor: The color of the node edges.
        node_alpha: The transparency of the nodes.
        edge_alpha: The transparency of the edges.
        node_size: The size of the nodes.
        node_shape: The shape of the nodes.
        edge_width: The width of the edges.
        node_edgewidth: The width of the edges around the nodes.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        if node_color is None:
            node_color = plotter.get_annotated_node_colors(
                cmap="gist_rainbow",
                min_scale=0.5,
                max_scale=1.0,
                scale_factor=0.5,
                nonenriched_color=nonenriched_color,
                nonenriched_alpha=nonenriched_alpha,
                random_seed=887,
            )

        plotter.plot_network(
            node_size=plotter.get_annotated_node_sizes(
                enriched_size=100, nonenriched_size=node_size
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
        ("white", "black", "blue", 250, 0.0, 1.0, 1.0, "^"),
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
    """Test plot_subnetwork with different node, edge, and node edge colors, sizes, edge widths, shapes, and alpha values.

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

    Returns:
        None
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
        (
            None,
            0.2,
            0.1,
            5,
            0.8,
            250,
            "solid",
            1.5,
        ),  # Test case 1: Annotated colors, alpha 0.2, fill_alpha 0.1
        (
            "red",
            0.5,
            0.3,
            6,
            1.0,
            300,
            "dashed",
            2.0,
        ),  # Test case 2: Red contours, alpha 0.5, fill_alpha 0.3, custom bandwidth and grid size
        (
            (0.2, 0.6, 0.8),
            0.3,
            0.15,
            4,
            0.6,
            200,
            "dotted",
            1.0,
        ),  # Test case 3: Light blue (RGB), alpha 0.3, fill_alpha 0.15, dotted line
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
    """Test plot_contours with different color formats, alpha values, fill_alpha values, contour levels, bandwidths, grid sizes, and line styles.

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

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        if color is None:
            color = plotter.get_annotated_contour_colors(cmap="gist_rainbow", random_seed=887)

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
            0.5,
            0.2,
            5,
            0.8,
            250,
            "solid",
            1.5,
        ),  # Test case 1: Red with alpha 0.5, fill_alpha 0.2, solid line
        (
            (0.2, 0.6, 0.8),
            0.3,
            0.1,
            4,
            1.0,
            300,
            "dashed",
            2.0,
        ),  # Test case 2: Light blue RGB with alpha 0.3, fill_alpha 0.1, dashed line
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
    """Test plot_subcontour with different color formats, alpha values, fill_alpha values, contour levels, bandwidths, grid sizes, and line styles.

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

    Returns:
        None
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
    "fontcolor, arrow_color, font_alpha, arrow_alpha, min_words, max_words, min_word_length, max_word_length, max_labels, scale, offset, font, fontsize, arrow_linewidth, arrow_style, overlay_ids, ids_to_keep, ids_to_replace",
    [
        (
            None,
            None,
            1.0,
            1.0,
            2,
            4,
            3,
            10,
            10,
            1.25,
            0.10,
            "Arial",
            10,
            1,
            "->",
            False,
            None,
            None,
        ),  # Test case 1: Annotated label colors, full opacity, max labels 10, default arrow style
        (
            "red",
            "blue",
            0.5,
            0.5,
            1,
            5,
            4,
            15,
            5,
            1.5,
            0.15,
            "Helvetica",
            12,
            2,
            "-|>",
            True,
            ["LSM1", "LSM2"],
            None,
        ),  # Test case 2: Custom colors, semi-transparent, max labels 5, arrow_style "-|>", with overlay_ids and ids_to_keep
        (
            (0.2, 0.6, 0.8),
            (1.0, 0.0, 0.0),
            0.3,
            0.3,
            3,
            6,
            2,
            20,
            8,
            1.35,
            0.12,
            "Times New Roman",
            14,
            1.5,
            "<|-",
            False,
            ["LSM1", "LSM3"],
            {"LSM3": "custom label"},
        ),  # Test case 3: Custom RGB colors, with alpha and word limits, arrow_style "<|-", max labels 8, ids_to_keep, and ids_to_replace
    ],
)
def test_plot_labels_with_custom_params(
    risk_obj,
    graph,
    fontcolor,
    arrow_color,
    font_alpha,
    arrow_alpha,
    min_words,
    max_words,
    min_word_length,
    max_word_length,
    max_labels,
    scale,
    offset,
    font,
    fontsize,
    arrow_linewidth,
    arrow_style,
    overlay_ids,
    ids_to_keep,
    ids_to_replace,
):
    """Test plot_labels with different label and arrow colors, alpha values, word constraints, max labels, various style parameters,
       and new params for overlay_ids, ids_to_keep, ids_to_replace, and arrow_style.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which labels will be plotted.
        fontcolor: The color of the label text (None for annotated, string, or RGB tuple).
        arrow_color: The color of the label arrows (None for annotated, string, or RGB tuple).
        font_alpha: The transparency of the label text.
        arrow_alpha: The transparency of the label arrows.
        min_words: Minimum number of words in the label.
        max_words: Maximum number of words in the label.
        min_word_length: Minimum length of each word.
        max_word_length: Maximum length of each word.
        max_labels: Maximum number of labels to display.
        scale: Scale factor for positioning labels around the perimeter.
        offset: Offset distance for labels from the perimeter.
        font: Font name for the labels.
        fontsize: Font size for the labels.
        arrow_linewidth: Line width for the arrows pointing to centroids.
        arrow_style: The style of the arrows (e.g., '->', '-|>', '<|-').
        overlay_ids: Whether to overlay the domain IDs in the center of the centroids.
        ids_to_keep: List of IDs to prioritize for labeling.
        ids_to_replace: Dictionary mapping domain IDs to custom labels.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        if fontcolor is None:
            fontcolor = plotter.get_annotated_label_colors(cmap="gist_rainbow", random_seed=887)

        if arrow_color is None:
            arrow_color = plotter.get_annotated_label_colors(cmap="gist_rainbow", random_seed=887)

        plotter.plot_labels(
            scale=scale,
            offset=offset,
            font=font,
            fontsize=fontsize,
            fontcolor=fontcolor,
            fontalpha=font_alpha,
            arrow_linewidth=arrow_linewidth,
            arrow_style=arrow_style,
            arrow_color=arrow_color,
            arrow_alpha=arrow_alpha,
            max_labels=max_labels,
            max_words=max_words,
            min_words=min_words,
            max_word_length=max_word_length,
            min_word_length=min_word_length,
            words_to_omit=["process", "biosynthetic"],
            overlay_ids=overlay_ids,
            ids_to_keep=ids_to_keep,
            ids_to_replace=ids_to_replace,
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "fontcolor, arrow_color, font_alpha, arrow_alpha, fontsize, radial_position, scale, offset, arrow_linewidth, font, arrow_style",
    [
        (
            "white",
            "white",
            1.0,
            1.0,
            14,
            73,
            1.6,
            0.10,
            1.5,
            "Arial",
            "->",
        ),  # Test case 1: Full opacity, white font and arrow, fontsize 14, radial position 73, default arrow style
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
        ),  # Test case 2: Semi-transparent, red font, blue arrow, fontsize 16, radial position 120, custom arrow style
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
        ),  # Test case 3: Custom RGB colors, with alpha and fontsize 18, radial position 45, custom arrow style
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
):
    """Test plot_sublabel with different label and arrow colors, alpha values, label positioning, and other style parameters.

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

    Returns:
        None
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
        )
    finally:
        plt.close("all")
