"""
risk/network/graph
~~~~~~~~~~~~~~~~~~
"""

import networkx as nx
import numpy as np


def calculate_edge_lengths(G, include_edge_weight=False, compute_sphere=True, dimple_factor=None):
    # Normalize graph coordinates
    normalize_graph_coordinates(G)
    # Normalize weights
    normalize_weights(G)
    # Conditionally map nodes to a sphere based on `compute_sphere`
    if compute_sphere:
        map_to_sphere(G)
        # Identify subclusters - TODO: play with this value and research optimal radius! So far, 1 works best
        neighborhood_radius = np.pi / 2  # (4 * 1.0 (normalized diameter))
        partition = find_subclusters_with_shortest_path(G, neighborhood_radius)
        # This is key to offer more dynamic range for the user; dimple factors don't need to be large
        dimple_factor = 0.0 if dimple_factor is None else dimple_factor / 1000
        # Create dimples
        create_dimples(G, partition, dimple_factor=dimple_factor)

    for u, v, edge_data in G.edges(data=True):
        if compute_sphere:
            u_coords = np.array([G.nodes[u]["x"], G.nodes[u]["y"], G.nodes[u].get("z", 0)])
            v_coords = np.array([G.nodes[v]["x"], G.nodes[v]["y"], G.nodes[v].get("z", 0)])
            # Calculate the spherical distance
            dist = np.arccos(np.clip(np.dot(u_coords, v_coords), -1.0, 1.0))
        else:
            # If not computing sphere, use only x, y for planar distance
            u_coords = np.array([G.nodes[u]["x"], G.nodes[u]["y"]])
            v_coords = np.array([G.nodes[v]["x"], G.nodes[v]["y"]])
            # Calculate the planar distance
            dist = np.linalg.norm(u_coords - v_coords)

        if include_edge_weight and "normalized_weight" in edge_data:
            # Invert the weight influence such that higher weights bring nodes closer
            G.edges[u, v]["length"] = dist / (
                edge_data["normalized_weight"] + 10e-12  # Avoid division by zero
            )
        else:
            G.edges[u, v]["length"] = dist  # Use calculated distance directly

    return G


def map_to_sphere(G):
    # Extract x, y coordinates from the graph nodes
    xy_coords = np.array([[G.nodes[node]["x"], G.nodes[node]["y"]] for node in G.nodes()])

    # Normalize the coordinates between [0, 1]
    min_vals = np.min(xy_coords, axis=0)
    max_vals = np.max(xy_coords, axis=0)
    normalized_xy = (xy_coords - min_vals) / (max_vals - min_vals)

    # Map normalized coordinates to theta and phi on a sphere
    theta = normalized_xy[:, 0] * np.pi * 2
    phi = normalized_xy[:, 1] * np.pi

    # Convert spherical coordinates to Cartesian coordinates for 3D sphere
    for i, node in enumerate(G.nodes()):
        x = np.sin(phi[i]) * np.cos(theta[i])
        y = np.sin(phi[i]) * np.sin(theta[i])
        z = np.cos(phi[i])
        G.nodes[node]["x"] = x
        G.nodes[node]["y"] = y
        G.nodes[node]["z"] = z


def find_subclusters_with_shortest_path(G, neighborhood_radius):
    # Compute Djikstra's shortest path for each pair of nodes within a specified cutoff
    all_shortest_paths = dict(
        nx.all_pairs_dijkstra_path_length(
            G,
            weight="length",
            cutoff=neighborhood_radius,
        )
    )
    # Identify subclusters based on the shortest path lengths
    subclusters = {}
    for source, targets in all_shortest_paths.items():
        for target, length in targets.items():
            if length <= neighborhood_radius:
                if source not in subclusters:
                    subclusters[source] = set()
                subclusters[source].add(target)
    return subclusters


def create_dimples(G, subclusters, dimple_factor=0.20):
    # Create a strength metric for subclusters (here using size)
    subcluster_strengths = {node: len(neighbors) for node, neighbors in subclusters.items()}

    # Normalize the subcluster strengths and apply dimples
    max_strength = max(subcluster_strengths.values())
    for node, strength in subcluster_strengths.items():
        depth_factor = strength / max_strength * dimple_factor
        x, y, z = G.nodes[node]["x"], G.nodes[node]["y"], G.nodes[node]["z"]
        norm = np.sqrt(x**2 + y**2 + z**2)
        G.nodes[node]["z"] -= (z / norm) * depth_factor  # Adjust Z for a dimple


def normalize_graph_coordinates(G):
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


def normalize_weights(G):
    weights = [data["weight"] for _, _, data in G.edges(data=True) if "weight" in data]
    if weights:  # Ensure there are weighted edges
        min_weight = min(weights)
        max_weight = max(weights)
        range_weight = max_weight - min_weight if max_weight > min_weight else 1
        for u, v, data in G.edges(data=True):
            if "weight" in data:
                data["normalized_weight"] = (data["weight"] - min_weight) / range_weight
