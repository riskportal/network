def test_initialize_plotter(risk_obj, graph):
    plotter = initialize_plotter(risk_obj, graph)

    assert plotter is not None  # Ensure the plotter is initialized
    assert hasattr(plotter, "graph")  # Check that the plotter has a graph attribute


def test_plot_network(risk_obj, graph):
    plotter = initialize_plotter(risk_obj, graph)
    plot_network(plotter)

    # You can add more checks to ensure nodes and edges are correctly processed if possible
    assert plotter is not None


def test_plot_subnetwork(risk_obj, graph):
    plotter = initialize_plotter(risk_obj, graph)
    plot_subnetwork(plotter)

    assert plotter is not None


def test_plot_contours(risk_obj, graph):
    plotter = initialize_plotter(risk_obj, graph)
    plot_contours(plotter)

    assert plotter is not None


def test_plot_subcontour(risk_obj, graph):
    plotter = initialize_plotter(risk_obj, graph)
    plot_subcontour(plotter)

    assert plotter is not None


def test_plot_labels(risk_obj, graph):
    plotter = initialize_plotter(risk_obj, graph)
    plot_labels(plotter)

    assert plotter is not None


def test_plot_sublabel(risk_obj, graph):
    plotter = initialize_plotter(risk_obj, graph)
    plot_sublabel(plotter)

    assert plotter is not None


def test_display_plot(risk_obj, graph):
    plotter = initialize_plotter(risk_obj, graph)
    display_plot(plotter)

    assert plotter is not None


def initialize_plotter(risk, graph):
    return risk.load_plotter(
        graph=graph,
        figsize=(15, 15),
        background_color="black",
        plot_outline=True,
        outline_color="white",
        outline_scale=1.05,
    )


def plot_network(plotter):
    plotter.plot_network(
        node_size=plotter.get_annotated_node_sizes(enriched_nodesize=100, nonenriched_nodesize=10),
        edge_width=0.0,
        node_color=plotter.get_annotated_node_colors(
            cmap="gist_rainbow",
            min_scale=0.5,
            max_scale=1.0,
            nonenriched_color="white",
            random_seed=887,
        ),
        node_edgecolor="black",
        edge_color="white",
        node_shape="o",
    )


def plot_subnetwork(plotter):
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
        edge_width=0.0,
        node_color="white",
        node_edgecolor="black",
        edge_color="white",
        node_shape="^",
    )


def plot_contours(plotter):
    plotter.plot_contours(
        levels=5,
        bandwidth=0.8,
        grid_size=250,
        alpha=0.2,
        color=plotter.get_annotated_contour_colors(cmap="gist_rainbow", random_seed=887),
    )


def plot_subcontour(plotter):
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


def plot_labels(plotter):
    plotter.plot_labels(
        perimeter_scale=1.25,
        offset=0.10,
        font="Arial",
        fontsize=10,
        fontcolor=plotter.get_annotated_label_colors(cmap="gist_rainbow", random_seed=887),
        arrow_linewidth=1,
        arrow_color=plotter.get_annotated_label_colors(cmap="gist_rainbow", random_seed=887),
        max_words=4,
        min_words=2,
    )


def plot_sublabel(plotter):
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
    )


def display_plot(plotter):
    plotter.show()
