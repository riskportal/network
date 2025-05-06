"""
risk/_neighborhoods/_community
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import community as community_louvain
import igraph as ig
import markov_clustering as mc
import networkx as nx
import numpy as np
from leidenalg import RBConfigurationVertexPartition, find_partition
from networkx.algorithms.community import greedy_modularity_communities
from scipy.sparse import csr_matrix

from .._log import logger


def calculate_greedy_modularity_neighborhoods(
    network: nx.Graph, fraction_shortest_edges: float = 1.0
) -> csr_matrix:
    """
    Calculate neighborhoods using the Greedy Modularity method with CSR matrix output.

    Args:
        network (nx.Graph): The network graph.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering. Defaults to 1.0.

    Returns:
        csr_matrix: A binary neighborhood matrix (CSR) where nodes in the same community have 1, and others have 0.

    Raises:
        ValueError: If the subgraph has no edges after filtering.
        Warning: If the resulting subgraph has no edges after filtering.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Detect communities using the Greedy Modularity method
    communities = greedy_modularity_communities(subnetwork)
    # Get the list of nodes in the original NetworkX graph
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Prepare data for CSR matrix
    row_indices = []
    col_indices = []
    for community in communities:
        mapped_indices = [node_index_map[node] for node in community]
        for i in mapped_indices:
            for j in mapped_indices:
                row_indices.append(i)
                col_indices.append(j)

    # Create a CSR matrix
    num_nodes = len(nodes)
    data = np.ones(len(row_indices), dtype=int)
    neighborhoods = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

    return neighborhoods


def calculate_label_propagation_neighborhoods(
    network: nx.Graph, fraction_shortest_edges: float = 1.0
) -> csr_matrix:
    """
    Apply Label Propagation to the network to detect communities.

    Args:
        network (nx.Graph): The network graph.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering. Defaults to 1.0.

    Returns:
        csr_matrix: A binary neighborhood matrix (CSR) on Label Propagation.

    Raises:
        ValueError: If the subgraph has no edges after filtering.
        Warning: If the resulting subgraph has no edges after filtering.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Apply Label Propagation for community detection
    communities = nx.algorithms.community.label_propagation.label_propagation_communities(
        subnetwork
    )
    # Get the list of nodes in the network
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Prepare data for CSR matrix
    row_indices = []
    col_indices = []
    # Assign neighborhoods based on community labels using the mapped indices
    for community in communities:
        mapped_indices = [node_index_map[node] for node in community]
        for i in mapped_indices:
            for j in mapped_indices:
                row_indices.append(i)
                col_indices.append(j)

    # Create a CSR matrix
    num_nodes = len(nodes)
    data = np.ones(len(row_indices), dtype=int)
    neighborhoods = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

    return neighborhoods


def calculate_leiden_neighborhoods(
    network: nx.Graph,
    resolution: float = 1.0,
    fraction_shortest_edges: float = 1.0,
    random_seed: int = 888,
) -> csr_matrix:
    """
    Calculate neighborhoods using the Leiden method with CSR matrix output.

    Args:
        network (nx.Graph): The network graph.
        resolution (float, optional): Resolution parameter for the Leiden method. Defaults to 1.0.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering. Defaults to 1.0.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 888.

    Returns:
        csr_matrix: A binary neighborhood matrix (CSR) where nodes in the same community have 1, and others have 0.

    Raises:
        ValueError: If the subgraph has no edges after filtering.
        Warning: If the resulting subgraph has no edges after filtering.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Convert NetworkX graph to iGraph
    igraph_network = ig.Graph.from_networkx(subnetwork)
    # Apply Leiden algorithm using RBConfigurationVertexPartition, which supports resolution
    partition = find_partition(
        igraph_network,
        partition_type=RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_seed,
    )
    # Get the list of nodes in the original NetworkX graph
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Prepare data for CSR matrix
    row_indices = []
    col_indices = []
    for community in partition:
        mapped_indices = [node_index_map[igraph_network.vs[node]["_nx_name"]] for node in community]
        for i in mapped_indices:
            for j in mapped_indices:
                row_indices.append(i)
                col_indices.append(j)

    # Create a CSR matrix
    num_nodes = len(nodes)
    data = np.ones(len(row_indices), dtype=int)
    neighborhoods = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

    return neighborhoods


def calculate_louvain_neighborhoods(
    network: nx.Graph,
    resolution: float = 0.1,
    fraction_shortest_edges: float = 1.0,
    random_seed: int = 888,
) -> csr_matrix:
    """
    Calculate neighborhoods using the Louvain method.

    Args:
        network (nx.Graph): The network graph.
        resolution (float, optional): Resolution parameter for the Louvain method. Defaults to 0.1.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering. Defaults to 1.0.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 888.

    Returns:
        csr_matrix: A binary neighborhood matrix in CSR format.

    Raises:
        ValueError: If the subgraph has no edges after filtering.
        Warning: If the resulting subgraph has no edges after filtering.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Apply Louvain method to partition the network
    partition = community_louvain.best_partition(
        subnetwork, resolution=resolution, random_state=random_seed
    )
    # Get the list of nodes in the network and create a mapping to indices
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Group nodes by community
    community_groups = {}
    for node, community in partition.items():
        community_groups.setdefault(community, []).append(node)

    # Prepare data for CSR matrix
    row_indices = []
    col_indices = []
    for community_nodes in community_groups.values():
        mapped_indices = [node_index_map[node] for node in community_nodes]
        for i in mapped_indices:
            for j in mapped_indices:
                row_indices.append(i)
                col_indices.append(j)

    # Create a CSR matrix
    num_nodes = len(nodes)
    data = np.ones(len(row_indices), dtype=int)
    neighborhoods = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

    return neighborhoods


def calculate_markov_clustering_neighborhoods(
    network: nx.Graph, fraction_shortest_edges: float = 1.0
) -> csr_matrix:
    """
    Apply Markov Clustering (MCL) to the network and return a binary neighborhood matrix (CSR).

    Args:
        network (nx.Graph): The network graph.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering. Defaults to 1.0.

    Returns:
        csr_matrix: A binary neighborhood matrix (CSR) on Markov Clustering.

    Raises:
        ValueError: If the subgraph has no edges after filtering.
        RuntimeError: If MCL fails to run.
        Warning: If the resulting subgraph has no edges after filtering.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Check if the subgraph has edges
    if subnetwork.number_of_edges() == 0:
        raise ValueError("The subgraph has no edges. Adjust the fraction_shortest_edges parameter.")

    # Step 1: Convert the subnetwork to an adjacency matrix (CSR)
    subnetwork_nodes = list(subnetwork.nodes())
    adjacency_matrix = nx.to_scipy_sparse_array(subnetwork, nodelist=subnetwork_nodes)
    # Ensure the adjacency matrix is valid
    if adjacency_matrix.shape[0] == 0 or adjacency_matrix.shape[1] == 0:
        raise ValueError(
            "The adjacency matrix is empty. Check the input graph or filtering criteria."
        )

    # Convert the sparse matrix to dense format for MCL
    dense_matrix = adjacency_matrix.toarray()
    # Step 2: Run Markov Clustering (MCL) on the dense adjacency matrix
    try:
        result = mc.run_mcl(dense_matrix)
    except Exception as e:
        raise RuntimeError(f"Markov Clustering failed: {e}")

    clusters = mc.get_clusters(result)
    # Step 3: Prepare the original network nodes and indices
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    # Step 4: Prepare data for CSR matrix
    row_indices = []
    col_indices = []
    for cluster in clusters:
        for node_i in cluster:
            for node_j in cluster:
                # Map the indices back to the original network's node indices
                original_node_i = subnetwork_nodes[node_i]
                original_node_j = subnetwork_nodes[node_j]
                if original_node_i in node_index_map and original_node_j in node_index_map:
                    idx_i = node_index_map[original_node_i]
                    idx_j = node_index_map[original_node_j]
                    row_indices.append(idx_i)
                    col_indices.append(idx_j)

    # Step 5: Create a CSR matrix
    data = np.ones(len(row_indices), dtype=int)
    neighborhoods = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

    return neighborhoods


def calculate_spinglass_neighborhoods(
    network: nx.Graph, fraction_shortest_edges: float = 1.0
) -> csr_matrix:
    """
    Apply Spinglass Community Detection to the network, handling disconnected components.

    Args:
        network (nx.Graph): The network graph.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering. Defaults to 1.0.

    Returns:
        csr_matrix: A binary neighborhood matrix (CSR) based on Spinglass communities.

    Raises:
        ValueError: If the subgraph has no edges after filtering.
        Warning: If the resulting subgraph has no edges after filtering.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Step 1: Find connected components in the graph
    components = list(nx.connected_components(subnetwork))
    # Prepare data for CSR matrix
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    row_indices = []
    col_indices = []
    # Step 2: Run Spinglass on each connected component
    for component in components:
        # Extract the subgraph corresponding to the current component
        subgraph = network.subgraph(component)
        # Convert the subgraph to an iGraph object
        igraph_subgraph = ig.Graph.from_networkx(subgraph)
        # Ensure the subgraph is connected before running Spinglass
        if not igraph_subgraph.is_connected():
            logger.error("Warning: Subgraph is not connected. Skipping...")
            continue

        # Apply Spinglass community detection
        try:
            communities = igraph_subgraph.community_spinglass()
        except Exception as e:
            logger.error(f"Error running Spinglass on component: {e}")
            continue

        # Step 3: Assign neighborhoods based on community labels
        for community in communities:
            mapped_indices = [
                node_index_map[igraph_subgraph.vs[node]["_nx_name"]] for node in community
            ]
            for i in mapped_indices:
                for j in mapped_indices:
                    row_indices.append(i)
                    col_indices.append(j)

    # Step 4: Create a CSR matrix
    num_nodes = len(nodes)
    data = np.ones(len(row_indices), dtype=int)
    neighborhoods = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

    return neighborhoods


def calculate_walktrap_neighborhoods(
    network: nx.Graph, fraction_shortest_edges: float = 1.0
) -> csr_matrix:
    """
    Apply Walktrap Community Detection to the network with CSR matrix output.

    Args:
        network (nx.Graph): The network graph.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering. Defaults to 1.0.

    Returns:
        csr_matrix: A binary neighborhood matrix (CSR) on Walktrap communities.

    Raises:
        ValueError: If the subgraph has no edges after filtering.
        Warning: If the resulting subgraph has no edges after filtering.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Convert NetworkX graph to iGraph
    igraph_network = ig.Graph.from_networkx(subnetwork)
    # Apply Walktrap community detection
    communities = igraph_network.community_walktrap().as_clustering()
    # Get the list of nodes in the original NetworkX graph
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Prepare data for CSR matrix
    row_indices = []
    col_indices = []
    for community in communities:
        mapped_indices = [node_index_map[igraph_network.vs[node]["_nx_name"]] for node in community]
        for i in mapped_indices:
            for j in mapped_indices:
                row_indices.append(i)
                col_indices.append(j)

    # Create a CSR matrix
    num_nodes = len(nodes)
    data = np.ones(len(row_indices), dtype=int)
    neighborhoods = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

    return neighborhoods


def _create_percentile_limited_subgraph(G: nx.Graph, fraction_shortest_edges: float) -> nx.Graph:
    """
    Create a subgraph containing the shortest edges based on the specified rank fraction
    of all edge lengths in the input graph.

    Args:
        G (nx.Graph): The input graph with 'length' attributes on edges.
        fraction_shortest_edges (float): The rank fraction (between 0 and 1) to filter edges.

    Returns:
        nx.Graph: A subgraph with nodes and edges where the edges are within the shortest
        specified rank fraction.

    Raises:
        ValueError: If no edges with 'length' attributes are found in the graph.
        Warning: If the resulting subgraph has no edges after filtering.
    """
    # Step 1: Extract edges with their lengths
    edges_with_length = [(u, v, d) for u, v, d in G.edges(data=True) if "length" in d]
    if not edges_with_length:
        raise ValueError(
            "No edge lengths found in the graph. Ensure edges have 'length' attributes."
        )

    # Step 2: Sort edges by length in ascending order
    edges_with_length.sort(key=lambda x: x[2]["length"])
    # Step 3: Calculate the cutoff index for the given rank fraction
    cutoff_index = int(fraction_shortest_edges * len(edges_with_length))
    if cutoff_index == 0:
        raise ValueError("The rank fraction is too low, resulting in no edges being included.")

    # Step 4: Create the subgraph by selecting only the shortest edges within the rank fraction
    subgraph = nx.Graph()
    subgraph.add_nodes_from(G.nodes(data=True))  # Retain all nodes from the original graph
    subgraph.add_edges_from(edges_with_length[:cutoff_index])
    # Step 5: Remove nodes with no edges
    subgraph.remove_nodes_from(list(nx.isolates(subgraph)))
    # Step 6: Check if the resulting subgraph has no edges and issue a warning
    if subgraph.number_of_edges() == 0:
        raise Warning("The resulting subgraph has no edges. Consider adjusting the rank fraction.")

    return subgraph
