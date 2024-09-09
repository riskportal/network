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


def plot_network(plotter):
    """Plot the full network using the plotter.

    Args:
        plotter: The initialized plotter object.

    Returns:
        None
    """
    try:
        # Optional: Plot network nodes and edges
        plotter.plot_network(
            node_size=plotter.get_annotated_node_sizes(
                enriched_nodesize=100, nonenriched_nodesize=25
            ),
            node_shape="o",  # OPTIONS: Circle shape
            edge_width=0.0,
            node_color=plotter.get_annotated_node_colors(
                cmap="gist_rainbow",
                min_scale=0.25,
                max_scale=1.0,
                scale_factor=0.5,
                alpha=1.0,  # Alpha for enriched nodes
                nonenriched_color="white",
                nonenriched_alpha=0.1,  # Alpha for non-enriched nodes
                random_seed=887,
            ),
            node_edgecolor="black",
            edge_color="white",
            node_alpha=0.1,  # This will override the alpha value in node_color if set
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
            node_shape="^",  # Triangle up for subnetwork
            edge_width=0.5,  # Optional: Adjust edge width for subnetwork
            node_color="skyblue",  # Subnetwork-specific node color
            node_edgecolor="black",
            edge_color="white",
            node_alpha=0.5,  # Semi-transparent nodes
            edge_alpha=0.5,  # Semi-transparent edges
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
            perimeter_scale=1.25,
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
            perimeter_scale=1.6,
            offset=0.10,
            font="Arial",
            fontsize=14,
            fontcolor="white",
            arrow_linewidth=1.5,
            arrow_color="white",
            fontalpha=0.5,  # Added transparency for font
            arrow_alpha=0.5,  # Added transparency for arrows
        )
    finally:
        plt.close("all")


# NOTE: Displaying plots during testing can cause the program to hang. Avoid including plot displays in tests.
# Now, let's test the plotter with different custom settings.


@pytest.mark.parametrize(
    "color, alpha",
    [
        ("white", 1.0),  # Test case 1: white color, full opacity
        ((0.5, 0.8, 1.0), 0.5),  # Test case 2: light blue (RGB), semi-transparent
    ],
)
def test_plot_border_with_custom_color_and_alpha(risk_obj, graph, color, alpha):
    """Test plot_border with different color and alpha parameters.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
        color: The color parameter for the border.
        alpha: The transparency parameter for the border.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        plotter.plot_border(
            scale=1.05,
            linestyle="solid",
            linewidth=1.5,
            color=color,
            alpha=alpha,
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "node_color, nonenriched_color, nonenriched_alpha, edge_color, node_edgecolor, node_alpha, edge_alpha",
    [
        (
            None,
            "white",
            0.1,  # Test case 1: non-enriched nodes have alpha 0.1
            "black",
            "blue",
            1.0,
            1.0,
        ),  # Test case 1: Annotated node colors, white non-enriched, black edges, blue node edge, full opacity
        (
            (0.2, 0.6, 0.8),
            (1.0, 1.0, 0.5),
            0.5,  # Test case 2: non-enriched nodes have alpha 0.5
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            0.5,
            0.5,
        ),  # Test case 2: Custom node colors, yellow non-enriched, red edges, green node edge, 50% opacity
    ],
)
def test_plot_network_with_custom_colors_and_alpha(
    risk_obj,
    graph,
    node_color,
    nonenriched_color,
    nonenriched_alpha,
    edge_color,
    node_edgecolor,
    node_alpha,
    edge_alpha,
):
    """Test plot_network with different node, edge, and node edge colors and alpha values.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object to be plotted.
        plotter: The initialized plotter object.
        node_color: The color of the network nodes or None to use annotated colors.
        nonenriched_color: The color for non-enriched nodes.
        nonenriched_alpha: The transparency of the non-enriched nodes.
        edge_color: The color of the network edges.
        node_edgecolor: The color of the node edges.
        node_alpha: The transparency of the nodes.
        edge_alpha: The transparency of the edges.

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
                nonenriched_alpha=nonenriched_alpha,  # Using the nonenriched_alpha value
                random_seed=887,
            )

        plotter.plot_network(
            node_size=plotter.get_annotated_node_sizes(
                enriched_nodesize=100, nonenriched_nodesize=10
            ),
            edge_width=0.0,
            node_color=node_color,
            node_edgecolor=node_edgecolor,
            edge_color=edge_color,
            node_shape="o",
            node_alpha=node_alpha,
            edge_alpha=edge_alpha,
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "node_color, edge_color, node_edgecolor, node_size, edge_width, node_alpha, edge_alpha",
    [
        (
            "white",
            "black",
            "blue",
            250,
            0.0,
            1.0,
            1.0,
        ),  # Test case 1: Custom colors, node size 250, full opacity
        (
            (0.2, 0.6, 0.8),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            300,
            0.5,
            0.5,
            0.5,
        ),  # Test case 2: Custom colors, node size 300, semi-transparent
    ],
)
def test_plot_subnetwork_with_custom_colors_sizes_and_alpha(
    risk_obj,
    graph,
    node_color,
    edge_color,
    node_edgecolor,
    node_size,
    edge_width,
    node_alpha,
    edge_alpha,
):
    """Test plot_subnetwork with different node, edge, and node edge colors, sizes, edge widths, and alpha values.

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

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
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
            node_size=node_size,
            edge_width=edge_width,
            node_color=node_color,
            node_edgecolor=node_edgecolor,
            edge_color=edge_color,
            node_shape="^",  # Keeping node shape constant
            node_alpha=node_alpha,  # Applying node transparency
            edge_alpha=edge_alpha,  # Applying edge transparency
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "color, alpha",
    [
        (None, 0.2),  # Test case 1: Use annotated contour colors with alpha 0.2
        ("red", 0.5),  # Test case 2: Use string color "red" with alpha 0.5
        ((0.2, 0.6, 0.8), 0.3),  # Test case 3: Use RGB value for light blue with alpha 0.3
    ],
)
def test_plot_contours_with_custom_colors(risk_obj, graph, color, alpha):
    """Test plot_contours with different color formats and alpha values.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which contours will be plotted.
        color: The color of the contours (None for annotated, string, or RGB tuple).
        alpha: The transparency of the contours.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
    try:
        if color is None:
            color = plotter.get_annotated_contour_colors(cmap="gist_rainbow", random_seed=887)

        plotter.plot_contours(
            levels=5,
            bandwidth=0.8,
            grid_size=250,
            alpha=alpha,
            color=color,
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "color, alpha",
    [
        ("red", 0.5),  # Test case 1: Use string color "red" with alpha 0.5
        ((0.2, 0.6, 0.8), 0.3),  # Test case 2: Use RGB value for light blue with alpha 0.3
    ],
)
def test_plot_subcontour_with_custom_colors(risk_obj, graph, color, alpha):
    """Test plot_subcontour with different color formats and alpha values.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which subcontours will be plotted.
        color: The color of the subcontours (None for annotated, string, or RGB tuple).
        alpha: The transparency of the subcontours.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
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
            alpha=alpha,
            color=color,
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "fontcolor, arrow_color, font_alpha, arrow_alpha, min_words, max_words, min_word_length, max_word_length, max_labels",
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
        ),  # Test case 1: Annotated label colors, full opacity, max labels 10
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
        ),  # Test case 2: Custom colors, semi-transparent, max labels 5
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
        ),  # Test case 3: Custom RGB colors, with alpha and word limits, max labels 8
    ],
)
def test_plot_labels_with_custom_colors_and_words(
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
):
    """Test plot_labels with different label and arrow colors, alpha values, word constraints, and max labels.

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
            perimeter_scale=1.25,
            offset=0.10,
            font="Arial",
            fontsize=10,
            fontcolor=fontcolor,
            fontalpha=font_alpha,
            arrow_linewidth=1,
            arrow_color=arrow_color,
            arrow_alpha=arrow_alpha,
            max_labels=max_labels,  # Adding the max_labels parameter
            max_words=max_words,
            min_words=min_words,
            max_word_length=max_word_length,
            min_word_length=min_word_length,
            words_to_omit=["process", "biosynthetic"],
        )
    finally:
        plt.close("all")


@pytest.mark.parametrize(
    "fontcolor, arrow_color, font_alpha, arrow_alpha, fontsize, radial_position",
    [
        (
            "white",
            "white",
            1.0,
            1.0,
            14,
            73,
        ),  # Test case 1: Full opacity, white font and arrow, fontsize 14, radial position 73
        (
            "red",
            "blue",
            0.5,
            0.5,
            16,
            120,
        ),  # Test case 2: Semi-transparent, red font, blue arrow, fontsize 16, radial position 120
        (
            (0.2, 0.6, 0.8),
            (1.0, 0.0, 0.0),
            0.8,
            0.7,
            18,
            45,
        ),  # Test case 3: Custom RGB colors, with alpha and fontsize 18, radial position 45
    ],
)
def test_plot_sublabel_with_custom_colors_and_alpha(
    risk_obj, graph, fontcolor, arrow_color, font_alpha, arrow_alpha, fontsize, radial_position
):
    """Test plot_sublabel with different label and arrow colors, alpha values, and label positioning.

    Args:
        risk_obj: The RISK object instance used for plotting.
        graph: The graph object on which the sublabel will be plotted.
        fontcolor: The color of the label text (string or RGB tuple).
        arrow_color: The color of the label arrows (string or RGB tuple).
        font_alpha: The transparency of the label text.
        arrow_alpha: The transparency of the label arrows.
        fontsize: The size of the label text.
        radial_position: The radial position of the label.

    Returns:
        None
    """
    plotter = initialize_plotter(risk_obj, graph)
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
            radial_position=radial_position,
            perimeter_scale=1.6,
            offset=0.10,
            font="Arial",
            fontsize=fontsize,
            fontcolor=fontcolor,
            fontalpha=font_alpha,  # Applying font transparency
            arrow_linewidth=1.5,
            arrow_color=arrow_color,
            arrow_alpha=arrow_alpha,  # Applying arrow transparency
        )
    finally:
        plt.close("all")
