"""
risk/network/geometry
~~~~~~~~~~~~~~~~~~~~~
"""

import networkx as nx
import numpy as np


def assign_edge_lengths(
    G: nx.Graph,
    compute_sphere: bool = True,
    surface_depth: float = 0.0,
) -> nx.Graph:
    """Assign edge lengths in the graph, optionally mapping nodes to a sphere.

    Args:
        G (nx.Graph): The input graph.
        compute_sphere (bool): Whether to map nodes to a sphere. Defaults to True.
        surface_depth (float): The surface depth for mapping to a sphere. Defaults to 0.0.

    Returns:
        nx.Graph: The graph with applied edge lengths.
    """

    def compute_distance_vectorized(coords, is_sphere):
        """Compute distances between pairs of coordinates."""
        u_coords, v_coords = coords[:, 0, :], coords[:, 1, :]
        if is_sphere:
            u_coords /= np.linalg.norm(u_coords, axis=1, keepdims=True)
            v_coords /= np.linalg.norm(v_coords, axis=1, keepdims=True)
            dot_products = np.einsum("ij,ij->i", u_coords, v_coords)
            return np.arccos(np.clip(dot_products, -1.0, 1.0))
        return np.linalg.norm(u_coords - v_coords, axis=1)

    # Normalize graph coordinates
    _normalize_graph_coordinates(G)

    # Map nodes to sphere and adjust depth if required
    if compute_sphere:
        _map_to_sphere(G)
        G_depth = _create_depth(G, surface_depth=surface_depth)
    else:
        G_depth = G

    # Precompute edge coordinate arrays and compute distances in bulk
    edge_data = np.array(
        [
            [
                np.array(
                    [G_depth.nodes[u]["x"], G_depth.nodes[u]["y"], G_depth.nodes[u].get("z", 0)]
                ),
                np.array(
                    [G_depth.nodes[v]["x"], G_depth.nodes[v]["y"], G_depth.nodes[v].get("z", 0)]
                ),
            ]
            for u, v in G_depth.edges
        ]
    )
    # Compute distances
    distances = compute_distance_vectorized(edge_data, compute_sphere)
    # Assign distances back to the graph
    for (u, v), distance in zip(G_depth.edges, distances):
        G.edges[u, v]["length"] = distance

    return G


def _map_to_sphere(G: nx.Graph) -> None:
    """Map the x and y coordinates of graph nodes onto a 3D sphere.

    Args:
        G (nx.Graph): The input graph with nodes having 'x' and 'y' coordinates.
    """
    # Extract x, y coordinates as a NumPy array
    nodes = list(G.nodes)
    xy_coords = np.array([[G.nodes[node]["x"], G.nodes[node]["y"]] for node in nodes])
    # Normalize coordinates between [0, 1]
    min_vals = xy_coords.min(axis=0)
    max_vals = xy_coords.max(axis=0)
    normalized_xy = (xy_coords - min_vals) / (max_vals - min_vals)
    # Convert normalized coordinates to spherical coordinates
    theta = normalized_xy[:, 0] * np.pi * 2
    phi = normalized_xy[:, 1] * np.pi
    # Compute 3D Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    # Assign coordinates back to graph nodes in bulk
    xyz_coords = {node: {"x": x[i], "y": y[i], "z": z[i]} for i, node in enumerate(nodes)}
    nx.set_node_attributes(G, xyz_coords)


def _normalize_graph_coordinates(G: nx.Graph) -> None:
    """Normalize the x and y coordinates of the nodes in the graph to the [0, 1] range.

    Args:
        G (nx.Graph): The input graph with nodes having 'x' and 'y' coordinates.
    """
    # Extract x, y coordinates from the graph nodes
    xy_coords = np.array([[G.nodes[node]["x"], G.nodes[node]["y"]] for node in G.nodes()])
    # Calculate min and max values for x and y
    min_vals = np.min(xy_coords, axis=0)
    max_vals = np.max(xy_coords, axis=0)
    # Normalize the coordinates to [0, 1]
    normalized_xy = (xy_coords - min_vals) / (max_vals - min_vals)
    # Update the node coordinates with the normalized values
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["x"], G.nodes[node]["y"] = normalized_xy[i]


def _create_depth(G: nx.Graph, surface_depth: float = 0.0) -> nx.Graph:
    """Adjust the 'z' attribute of each node based on the subcluster strengths and normalized surface depth.

    Args:
        G (nx.Graph): The input graph.
        surface_depth (float): The maximum surface depth to apply for the strongest subcluster.

    Returns:
        nx.Graph: The graph with adjusted 'z' attribute for each node.
    """
    if surface_depth >= 1.0:
        surface_depth -= 1e-6  # Cap the surface depth to prevent a value of 1.0

    # Compute subclusters as connected components
    connected_components = list(nx.connected_components(G))
    subcluster_strengths = {}
    max_strength = 0
    # Precompute strengths and track the maximum strength
    for component in connected_components:
        size = len(component)
        max_strength = max(max_strength, size)
        for node in component:
            subcluster_strengths[node] = size

    # Avoid repeated lookups and computations by pre-fetching node data
    nodes = list(G.nodes(data=True))
    node_updates = {}
    for node, attrs in nodes:
        strength = subcluster_strengths[node]
        normalized_surface_depth = (strength / max_strength) * surface_depth
        x, y, z = attrs["x"], attrs["y"], attrs["z"]
        norm = np.sqrt(x**2 + y**2 + z**2)
        adjusted_z = z - (z / norm) * normalized_surface_depth
        node_updates[node] = {"z": adjusted_z}

    # Batch update node attributes
    nx.set_node_attributes(G, node_updates)

    return G
