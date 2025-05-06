"""
risk/_neighborhoods/_api
~~~~~~~~~~~~~~~~~~~~~~~~
"""

import copy
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from .._log import log_header, logger, params
from ._neighborhoods import get_network_neighborhoods
from ._stats import (
    compute_binom_test,
    compute_chi2_test,
    compute_hypergeom_test,
    compute_permutation_test,
    compute_poisson_test,
    compute_zscore_test,
)


class NeighborhoodsAPI:
    """
    Handles the loading of statistical results and annotation significance for neighborhoods.

    The NeighborhoodsAPI class provides methods to load neighborhood results from statistical tests.
    """

    def __init__(self) -> None:
        pass

    def load_neighborhoods_binom(
        self,
        network: nx.Graph,
        annotation: Dict[str, Any],
        distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        leiden_resolution: float = 1.0,
        fraction_shortest_edges: Union[float, List, Tuple, np.ndarray] = 0.5,
        null_distribution: str = "network",
        random_seed: int = 888,
    ) -> Dict[str, Any]:
        """
        Load significant neighborhoods for the network using the binomial test.

        Args:
            network (nx.Graph): The network graph.
            annotation (Dict[str, Any]): The annotation associated with the network.
            distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use. Can be a string for one
                metric or a list/tuple/ndarray of metrics ('greedy_modularity', 'louvain', 'leiden', 'label_propagation',
                'markov_clustering', 'walktrap', 'spinglass'). Defaults to 'louvain'.
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            leiden_resolution (float, optional): Resolution parameter for Leiden clustering. Defaults to 1.0.
            fraction_shortest_edges (float, List, Tuple, or np.ndarray, optional): Shortest edge rank fraction threshold(s) for creating subgraphs.
                Can be a single float for one threshold or a list/tuple of floats corresponding to multiple thresholds.
                Defaults to 0.5.
            null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".
            random_seed (int, optional): Seed for random number generation. Defaults to 888.

        Returns:
            Dict[str, Any]: Computed significance of neighborhoods.
        """
        log_header("Running binomial test")
        # Compute neighborhood significance using the binomial test
        return self._load_neighborhoods_by_statistical_test(
            network=network,
            annotation=annotation,
            distance_metric=distance_metric,
            louvain_resolution=louvain_resolution,
            leiden_resolution=leiden_resolution,
            fraction_shortest_edges=fraction_shortest_edges,
            null_distribution=null_distribution,
            random_seed=random_seed,
            statistical_test_key="binom",
            statistical_test_function=compute_binom_test,
        )

    def load_neighborhoods_chi2(
        self,
        network: nx.Graph,
        annotation: Dict[str, Any],
        distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        leiden_resolution: float = 1.0,
        fraction_shortest_edges: Union[float, List, Tuple, np.ndarray] = 0.5,
        null_distribution: str = "network",
        random_seed: int = 888,
    ) -> Dict[str, Any]:
        """
        Load significant neighborhoods for the network using the chi-squared test.

        Args:
            network (nx.Graph): The network graph.
            annotation (Dict[str, Any]): The annotation associated with the network.
            distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use. Can be a string for one
                metric or a list/tuple/ndarray of metrics ('greedy_modularity', 'louvain', 'leiden', 'label_propagation',
                'markov_clustering', 'walktrap', 'spinglass'). Defaults to 'louvain'.
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            leiden_resolution (float, optional): Resolution parameter for Leiden clustering. Defaults to 1.0.
            fraction_shortest_edges (float, List, Tuple, or np.ndarray, optional): Shortest edge rank fraction threshold(s) for creating subgraphs.
                Can be a single float for one threshold or a list/tuple of floats corresponding to multiple thresholds.
                Defaults to 0.5.
            null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".
            random_seed (int, optional): Seed for random number generation. Defaults to 888.

        Returns:
            Dict[str, Any]: Computed significance of neighborhoods.
        """
        log_header("Running chi-squared test")
        # Compute neighborhood significance using the chi-squared test
        return self._load_neighborhoods_by_statistical_test(
            network=network,
            annotation=annotation,
            distance_metric=distance_metric,
            louvain_resolution=louvain_resolution,
            leiden_resolution=leiden_resolution,
            fraction_shortest_edges=fraction_shortest_edges,
            null_distribution=null_distribution,
            random_seed=random_seed,
            statistical_test_key="chi2",
            statistical_test_function=compute_chi2_test,
        )

    def load_neighborhoods_hypergeom(
        self,
        network: nx.Graph,
        annotation: Dict[str, Any],
        distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        leiden_resolution: float = 1.0,
        fraction_shortest_edges: Union[float, List, Tuple, np.ndarray] = 0.5,
        null_distribution: str = "network",
        random_seed: int = 888,
    ) -> Dict[str, Any]:
        """
        Load significant neighborhoods for the network using the hypergeometric test.

        Args:
            network (nx.Graph): The network graph.
            annotation (Dict[str, Any]): The annotation associated with the network.
            distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use. Can be a string for one
                metric or a list/tuple/ndarray of metrics ('greedy_modularity', 'louvain', 'leiden', 'label_propagation',
                'markov_clustering', 'walktrap', 'spinglass'). Defaults to 'louvain'.
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            leiden_resolution (float, optional): Resolution parameter for Leiden clustering. Defaults to 1.0.
            fraction_shortest_edges (float, List, Tuple, or np.ndarray, optional): Shortest edge rank fraction threshold(s) for creating subgraphs.
                Can be a single float for one threshold or a list/tuple of floats corresponding to multiple thresholds.
                Defaults to 0.5.
            null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".
            random_seed (int, optional): Seed for random number generation. Defaults to 888.

        Returns:
            Dict[str, Any]: Computed significance of neighborhoods.
        """
        log_header("Running hypergeometric test")
        # Compute neighborhood significance using the hypergeometric test
        return self._load_neighborhoods_by_statistical_test(
            network=network,
            annotation=annotation,
            distance_metric=distance_metric,
            louvain_resolution=louvain_resolution,
            leiden_resolution=leiden_resolution,
            fraction_shortest_edges=fraction_shortest_edges,
            null_distribution=null_distribution,
            random_seed=random_seed,
            statistical_test_key="hypergeom",
            statistical_test_function=compute_hypergeom_test,
        )

    def load_neighborhoods_permutation(
        self,
        network: nx.Graph,
        annotation: Dict[str, Any],
        distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        leiden_resolution: float = 1.0,
        fraction_shortest_edges: Union[float, List, Tuple, np.ndarray] = 0.5,
        score_metric: str = "sum",
        null_distribution: str = "network",
        num_permutations: int = 1000,
        random_seed: int = 888,
        max_workers: int = 1,
    ) -> Dict[str, Any]:
        """
        Load significant neighborhoods for the network using the permutation test.

        Args:
            network (nx.Graph): The network graph.
            annotation (Dict[str, Any]): The annotation associated with the network.
            distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use. Can be a string for one
                metric or a list/tuple/ndarray of metrics ('greedy_modularity', 'louvain', 'leiden', 'label_propagation',
                'markov_clustering', 'walktrap', 'spinglass'). Defaults to 'louvain'.
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            leiden_resolution (float, optional): Resolution parameter for Leiden clustering. Defaults to 1.0.
            fraction_shortest_edges (float, List, Tuple, or np.ndarray, optional): Shortest edge rank fraction threshold(s) for creating subgraphs.
                Can be a single float for one threshold or a list/tuple of floats corresponding to multiple thresholds.
                Defaults to 0.5.
            score_metric (str, optional): Scoring metric for neighborhood significance. Defaults to "sum".
            null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".
            num_permutations (int, optional): Number of permutations for significance testing. Defaults to 1000.
            random_seed (int, optional): Seed for random number generation. Defaults to 888.
            max_workers (int, optional): Maximum number of workers for parallel computation. Defaults to 1.

        Returns:
            Dict[str, Any]: Computed significance of neighborhoods.
        """
        log_header("Running permutation test")
        # Log and display permutation test settings, which is unique to this test
        logger.debug(f"Neighborhood scoring metric: '{score_metric}'")
        logger.debug(f"Number of permutations: {num_permutations}")
        logger.debug(f"Maximum workers: {max_workers}")
        # Compute neighborhood significance using the permutation test
        return self._load_neighborhoods_by_statistical_test(
            network=network,
            annotation=annotation,
            distance_metric=distance_metric,
            louvain_resolution=louvain_resolution,
            leiden_resolution=leiden_resolution,
            fraction_shortest_edges=fraction_shortest_edges,
            null_distribution=null_distribution,
            random_seed=random_seed,
            statistical_test_key="permutation",
            statistical_test_function=compute_permutation_test,
            score_metric=score_metric,
            num_permutations=num_permutations,
            max_workers=max_workers,
        )

    def load_neighborhoods_poisson(
        self,
        network: nx.Graph,
        annotation: Dict[str, Any],
        distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        leiden_resolution: float = 1.0,
        fraction_shortest_edges: Union[float, List, Tuple, np.ndarray] = 0.5,
        null_distribution: str = "network",
        random_seed: int = 888,
    ) -> Dict[str, Any]:
        """
        Load significant neighborhoods for the network using the Poisson test.

        Args:
            network (nx.Graph): The network graph.
            annotation (Dict[str, Any]): The annotation associated with the network.
            distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use. Can be a string for one
                metric or a list/tuple/ndarray of metrics ('greedy_modularity', 'louvain', 'leiden', 'label_propagation',
                'markov_clustering', 'walktrap', 'spinglass'). Defaults to 'louvain'.
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            leiden_resolution (float, optional): Resolution parameter for Leiden clustering. Defaults to 1.0.
            fraction_shortest_edges (float, List, Tuple, or np.ndarray, optional): Shortest edge rank fraction threshold(s) for creating subgraphs.
                Can be a single float for one threshold or a list/tuple of floats corresponding to multiple thresholds.
                Defaults to 0.5.
            null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".
            random_seed (int, optional): Seed for random number generation. Defaults to 888.

        Returns:
            Dict[str, Any]: Computed significance of neighborhoods.
        """
        log_header("Running Poisson test")
        # Compute neighborhood significance using the Poisson test
        return self._load_neighborhoods_by_statistical_test(
            network=network,
            annotation=annotation,
            distance_metric=distance_metric,
            louvain_resolution=louvain_resolution,
            leiden_resolution=leiden_resolution,
            fraction_shortest_edges=fraction_shortest_edges,
            null_distribution=null_distribution,
            random_seed=random_seed,
            statistical_test_key="poisson",
            statistical_test_function=compute_poisson_test,
        )

    def load_neighborhoods_zscore(
        self,
        network: nx.Graph,
        annotation: Dict[str, Any],
        distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        leiden_resolution: float = 1.0,
        fraction_shortest_edges: Union[float, List, Tuple, np.ndarray] = 0.5,
        null_distribution: str = "network",
        random_seed: int = 888,
    ) -> Dict[str, Any]:
        """
        Load significant neighborhoods for the network using the z-score test.

        Args:
            network (nx.Graph): The network graph.
            annotation (Dict[str, Any]): The annotation associated with the network.
            distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use. Can be a string for one
                metric or a list/tuple/ndarray of metrics ('greedy_modularity', 'louvain', 'leiden', 'label_propagation',
                'markov_clustering', 'walktrap', 'spinglass'). Defaults to 'louvain'.
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            leiden_resolution (float, optional): Resolution parameter for Leiden clustering. Defaults to 1.0.
            fraction_shortest_edges (float, List, Tuple, or np.ndarray, optional): Shortest edge rank fraction threshold(s) for creating subgraphs.
                Can be a single float for one threshold or a list/tuple of floats corresponding to multiple thresholds.
                Defaults to 0.5.
            null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".
            random_seed (int, optional): Seed for random number generation. Defaults to 888.

        Returns:
            Dict[str, Any]: Computed significance of neighborhoods.
        """
        log_header("Running z-score test")
        # Compute neighborhood significance using the z-score test
        return self._load_neighborhoods_by_statistical_test(
            network=network,
            annotation=annotation,
            distance_metric=distance_metric,
            louvain_resolution=louvain_resolution,
            leiden_resolution=leiden_resolution,
            fraction_shortest_edges=fraction_shortest_edges,
            null_distribution=null_distribution,
            random_seed=random_seed,
            statistical_test_key="zscore",
            statistical_test_function=compute_zscore_test,
        )

    def _load_neighborhoods_by_statistical_test(
        self,
        network: nx.Graph,
        annotation: Dict[str, Any],
        distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        leiden_resolution: float = 1.0,
        fraction_shortest_edges: Union[float, List, Tuple, np.ndarray] = 0.5,
        null_distribution: str = "network",
        random_seed: int = 888,
        statistical_test_key: str = "hypergeom",
        statistical_test_function: Any = compute_hypergeom_test,
        **kwargs,
    ):
        """
        Load and compute significant neighborhoods for the network using a specified statistical test.

        Args:
            network (nx.Graph): The input network graph.
            annotation (Dict[str, Any]): Annotation data associated with the network, including a "matrix" key with annotation values.
            distance_metric (Union[str, List, Tuple, np.ndarray], optional): The distance metric or clustering method to define neighborhoods.
                Can be a string specifying one method (e.g., 'louvain', 'leiden') or a collection of methods.
                Defaults to "louvain".
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            leiden_resolution (float, optional): Resolution parameter for Leiden clustering. Defaults to 1.0.
            fraction_shortest_edges (Union[float, List, Tuple, np.ndarray], optional): Fraction of shortest edges to consider for creating subgraphs.
                Can be a single value or a collection of thresholds for flexibility. Defaults to 0.5.
            null_distribution (str, optional): The type of null distribution to use ('network' or 'annotation').
                Defaults to "network".
            random_seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 888.
            statistical_test_key (str, optional): Key or name of the statistical test to be applied (e.g., "hypergeom", "poisson").
                Used for logging and debugging. Defaults to "hypergeom".
            statistical_test_function (Any, optional): The function implementing the statistical test.
                It should accept neighborhoods, annotation, null distribution, and additional kwargs.
                Defaults to `compute_hypergeom_test`.
            **kwargs: Additional parameters to be passed to the statistical test function.

        Returns:
            Dict[str, Any]: A dictionary containing the computed significance values for neighborhoods.
        """
        # Log null distribution type
        logger.debug(f"Null distribution: '{null_distribution}'")
        # Log neighborhood analysis parameters
        params.log_neighborhoods(
            distance_metric=distance_metric,
            louvain_resolution=louvain_resolution,
            leiden_resolution=leiden_resolution,
            fraction_shortest_edges=fraction_shortest_edges,
            statistical_test_function=statistical_test_key,
            null_distribution=null_distribution,
            random_seed=random_seed,
            **kwargs,
        )

        # Make a copy of the network to avoid modifying the original
        network = copy.copy(network)
        # Load neighborhoods based on the network and distance metric
        neighborhoods = self._load_neighborhoods(
            network,
            distance_metric,
            louvain_resolution=louvain_resolution,
            leiden_resolution=leiden_resolution,
            fraction_shortest_edges=fraction_shortest_edges,
            random_seed=random_seed,
        )
        # Apply statistical test function to compute neighborhood significance
        neighborhood_significance = statistical_test_function(
            neighborhoods=neighborhoods,
            annotation=annotation["matrix"],
            null_distribution=null_distribution,
            **kwargs,
        )

        # Return the computed neighborhood significance
        return neighborhood_significance

    def _load_neighborhoods(
        self,
        network: nx.Graph,
        distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        leiden_resolution: float = 1.0,
        fraction_shortest_edges: Union[float, List, Tuple, np.ndarray] = 0.5,
        random_seed: int = 888,
    ) -> csr_matrix:
        """
        Load significant neighborhoods for the network.

        Args:
            network (nx.Graph): The network graph.
            distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use. Can be a string for one
                metric or a list/tuple/ndarray of metrics ('greedy_modularity', 'louvain', 'leiden', 'label_propagation',
                'markov_clustering', 'walktrap', 'spinglass'). Defaults to 'louvain'.
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            leiden_resolution (float, optional): Resolution parameter for Leiden clustering. Defaults to 1.0.
            fraction_shortest_edges (float, List, Tuple, or np.ndarray, optional): Shortest edge rank fraction threshold(s) for creating subgraphs.
                Can be a single float for one threshold or a list/tuple of floats corresponding to multiple thresholds.
                Defaults to 0.5.
            random_seed (int, optional): Seed for random number generation. Defaults to 888.

        Returns:
            csr_matrix: Sparse neighborhood matrix calculated based on the selected distance metric.
        """
        # Display the chosen distance metric
        if distance_metric == "louvain":
            for_print_distance_metric = f"louvain (resolution={louvain_resolution})"
        elif distance_metric == "leiden":
            for_print_distance_metric = f"leiden (resolution={leiden_resolution})"
        else:
            for_print_distance_metric = distance_metric

        # Log and display neighborhood settings
        logger.debug(f"Distance metric: '{for_print_distance_metric}'")
        logger.debug(f"Edge length threshold: {fraction_shortest_edges}")
        logger.debug(f"Random seed: {random_seed}")

        # Compute neighborhoods
        neighborhoods = get_network_neighborhoods(
            network,
            distance_metric,
            fraction_shortest_edges,
            louvain_resolution=louvain_resolution,
            leiden_resolution=leiden_resolution,
            random_seed=random_seed,
        )

        # Return the sparse neighborhood matrix
        return neighborhoods
