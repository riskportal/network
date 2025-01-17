"""
risk/network/geometry
~~~~~~~~~~~~~~~~~~~~~
"""

import copy

import networkx as nx
import numpy as np


def assign_edge_lengths(
    G: nx.Graph,
    compute_sphere: bool = True,
    surface_depth: float = 0.0,
    include_edge_weight: bool = False,
) -> nx.Graph:
    """Assign edge lengths in the graph, optionally mapping nodes to a sphere and including edge weights.

    Args:
        G (nx.Graph): The input graph.
        compute_sphere (bool): Whether to map nodes to a sphere. Defaults to True.
        surface_depth (float): The surface depth for mapping to a sphere. Defaults to 0.0.
        include_edge_weight (bool): Whether to include edge weights in the calculation. Defaults to False.

    Returns:
        nx.Graph: The graph with applied edge lengths.
    """

    def compute_distance(
        u_coords: np.ndarray, v_coords: np.ndarray, is_sphere: bool = False
    ) -> float:
        """Compute the distance between two coordinate vectors.

        Args:
            u_coords (np.ndarray): Coordinates of the first point.
            v_coords (np.ndarray): Coordinates of the second point.
            is_sphere (bool, optional): If True, compute spherical distance. Defaults to False.

        Returns:
            float: The computed distance between the two points.
        """
        if is_sphere:
            # Normalize vectors and compute spherical distance using the dot product
            u_coords /= np.linalg.norm(u_coords)
            v_coords /= np.linalg.norm(v_coords)
            return np.arccos(np.clip(np.dot(u_coords, v_coords), -1.0, 1.0))
        else:
            # Compute Euclidean distance
            return np.linalg.norm(u_coords - v_coords)

    # Normalize graph coordinates
    _normalize_graph_coordinates(G)
    # Normalize weights
    _normalize_weights(G)
    # Use G_depth for edge length calculation
    if compute_sphere:
        # Map to sphere and adjust depth
        _map_to_sphere(G)
        G_depth = _create_depth(copy.deepcopy(G), surface_depth=surface_depth)
    else:
        # Calculate edge lengths directly on the plane
        G_depth = copy.deepcopy(G)

    for u, v, _ in G_depth.edges(data=True):
        u_coords = np.array([G_depth.nodes[u]["x"], G_depth.nodes[u]["y"]])
        v_coords = np.array([G_depth.nodes[v]["x"], G_depth.nodes[v]["y"]])
        if compute_sphere:
            u_coords = np.append(u_coords, G_depth.nodes[u].get("z", 0))
            v_coords = np.append(v_coords, G_depth.nodes[v].get("z", 0))

        distance = compute_distance(u_coords, v_coords, is_sphere=compute_sphere)
        # Assign edge lengths to the original graph
        if include_edge_weight:
            # Square root of the normalized weight is used to minimize the effect of large weights
            G.edges[u, v]["length"] = distance / np.sqrt(G.edges[u, v]["normalized_weight"] + 1e-6)
        else:
            # Use calculated distance directly
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


def _normalize_weights(G: nx.Graph) -> None:
    """Normalize the weights of the edges in the graph.

    Args:
        G (nx.Graph): The input graph with weighted edges.
    """
    # "weight" is present for all edges - weights are 1.0 if weight was not specified by the user
    weights = [data["weight"] for _, _, data in G.edges(data=True)]
    if weights:  # Ensure there are weighted edges
        min_weight = min(weights)
        max_weight = max(weights)
        range_weight = max_weight - min_weight if max_weight > min_weight else 1
        for _, _, data in G.edges(data=True):
            data["normalized_weight"] = (data["weight"] - min_weight) / range_weight


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
