"""
risk/neighborhoods/domains
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from contextlib import suppress
from itertools import product
from tqdm import tqdm
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

from risk.annotations import get_weighted_description
from risk.constants import GROUP_LINKAGE_METHODS, GROUP_DISTANCE_METRICS
from risk.log import logger


def define_domains(
    top_annotations: pd.DataFrame,
    significant_neighborhoods_significance: np.ndarray,
    linkage_criterion: str,
    linkage_method: str,
    linkage_metric: str,
) -> pd.DataFrame:
    """Define domains and assign nodes to these domains based on their significance scores and clustering,
    handling errors by assigning unique domains when clustering fails.

    Args:
        top_annotations (pd.DataFrame): DataFrame of top annotations data for the network nodes.
        significant_neighborhoods_significance (np.ndarray): The binary significance matrix below alpha.
        linkage_criterion (str): The clustering criterion for defining groups.
        linkage_method (str): The linkage method for clustering.
        linkage_metric (str): The linkage metric for clustering.

    Returns:
        pd.DataFrame: DataFrame with the primary domain for each node.
    """
    try:
        # Transpose the matrix to cluster annotations
        m = significant_neighborhoods_significance[:, top_annotations["significant_annotations"]].T
        best_linkage, best_metric, best_threshold = _optimize_silhouette_across_linkage_and_metrics(
            m, linkage_criterion, linkage_method, linkage_metric
        )
        # Perform hierarchical clustering
        Z = linkage(m, method=best_linkage, metric=best_metric)
        logger.warning(
            f"Linkage criterion: '{linkage_criterion}'\nLinkage method: '{best_linkage}'\nLinkage metric: '{best_metric}'"
        )
        logger.debug(f"Optimal linkage threshold: {round(best_threshold, 3)}")
        # Calculate the optimal threshold for clustering
        max_d_optimal = np.max(Z[:, 2]) * best_threshold
        # Assign domains to the annotations matrix
        domains = fcluster(Z, max_d_optimal, criterion=linkage_criterion)
        top_annotations["domain"] = 0
        top_annotations.loc[top_annotations["significant_annotations"], "domain"] = domains
    except ValueError:
        # If a ValueError is encountered, handle it by assigning unique domains
        n_rows = len(top_annotations)
        logger.error(
            f"Error encountered. Skipping clustering and assigning {n_rows} unique domains."
        )
        top_annotations["domain"] = range(1, n_rows + 1)  # Assign unique domains

    # Create DataFrames to store domain information
    node_to_significance = pd.DataFrame(
        data=significant_neighborhoods_significance,
        columns=[top_annotations.index.values, top_annotations["domain"]],
    )
    node_to_domain = node_to_significance.groupby(level="domain", axis=1).sum()

    # Find the maximum significance score for each node
    t_max = node_to_domain.loc[:, 1:].max(axis=1)
    t_idxmax = node_to_domain.loc[:, 1:].idxmax(axis=1)
    t_idxmax[t_max == 0] = 0

    # Assign all domains where the score is greater than 0
    node_to_domain["all_domains"] = node_to_domain.loc[:, 1:].apply(
        lambda row: list(row[row > 0].index), axis=1
    )
    # Assign primary domain
    node_to_domain["primary_domain"] = t_idxmax

    return node_to_domain


def trim_domains(
    domains: pd.DataFrame,
    top_annotations: pd.DataFrame,
    min_cluster_size: int = 5,
    max_cluster_size: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trim domains that do not meet size criteria and find outliers.

    Args:
        domains (pd.DataFrame): DataFrame of domain data for the network nodes.
        top_annotations (pd.DataFrame): DataFrame of top annotations data for the network nodes.
        min_cluster_size (int, optional): Minimum size of a cluster to be retained. Defaults to 5.
        max_cluster_size (int, optional): Maximum size of a cluster to be retained. Defaults to 1000.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Trimmed domains (pd.DataFrame)
            - A DataFrame with domain labels (pd.DataFrame)
    """
    # Identify domains to remove based on size criteria
    domain_counts = domains["primary_domain"].value_counts()
    to_remove = set(
        domain_counts[(domain_counts < min_cluster_size) | (domain_counts > max_cluster_size)].index
    )

    # Add invalid domain IDs
    invalid_domain_id = 888888
    invalid_domain_ids = {0, invalid_domain_id}
    # Mark domains to be removed
    top_annotations["domain"].replace(to_remove, invalid_domain_id, inplace=True)
    domains.loc[domains["primary_domain"].isin(to_remove), ["primary_domain"]] = invalid_domain_id

    # Normalize "num significant neighborhoods" by percentile for each domain and scale to 0-10
    top_annotations["normalized_value"] = top_annotations.groupby("domain")[
        "significant_neighborhood_significance_sums"
    ].transform(lambda x: (x.rank(pct=True) * 10).apply(np.ceil).astype(int))
    # Modify the lambda function to pass both full_terms and significant_significance_score
    top_annotations["combined_terms"] = top_annotations.apply(
        lambda row: " ".join([str(row["full_terms"])] * row["normalized_value"]), axis=1
    )

    # Perform the groupby operation while retaining the other columns and adding the weighting with significance scores
    domain_labels = (
        top_annotations.groupby("domain")
        .agg(
            full_terms=("full_terms", lambda x: list(x)),
            significance_scores=("significant_significance_score", lambda x: list(x)),
        )
        .reset_index()
    )
    domain_labels["combined_terms"] = domain_labels.apply(
        lambda row: get_weighted_description(
            pd.Series(row["full_terms"]), pd.Series(row["significance_scores"])
        ),
        axis=1,
    )

    # Rename the columns as necessary
    trimmed_domains_matrix = domain_labels.rename(
        columns={
            "domain": "id",
            "combined_terms": "normalized_description",
            "full_terms": "full_descriptions",
            "significance_scores": "significance_scores",
        }
    ).set_index("id")

    # Remove invalid domains
    valid_domains = domains[~domains["primary_domain"].isin(invalid_domain_ids)]
    valid_trimmed_domains_matrix = trimmed_domains_matrix[
        ~trimmed_domains_matrix.index.isin(invalid_domain_ids)
    ]
    return valid_domains, valid_trimmed_domains_matrix


def _optimize_silhouette_across_linkage_and_metrics(
    m: np.ndarray, linkage_criterion: str, linkage_method: str, linkage_metric: str
) -> Tuple[str, str, float]:
    """Optimize silhouette score across different linkage methods and distance metrics.

    Args:
        m (np.ndarray): Data matrix.
        linkage_criterion (str): Clustering criterion.
        linkage_method (str): Linkage method for clustering.
        linkage_metric (str): Linkage metric for clustering.

    Returns:
        Tuple[str, str, float]:
            - Best linkage method (str)
            - Best linkage metric (str)
            - Best threshold (float)
    """
    best_overall_method = linkage_method
    best_overall_metric = linkage_metric
    best_overall_score = -np.inf
    best_overall_threshold = 1

    linkage_methods = GROUP_LINKAGE_METHODS if linkage_method == "auto" else [linkage_method]
    linkage_metrics = GROUP_DISTANCE_METRICS if linkage_metric == "auto" else [linkage_metric]
    total_combinations = len(linkage_methods) * len(linkage_metrics)

    # Evaluating optimal linkage method and metric
    for method, metric in tqdm(
        product(linkage_methods, linkage_metrics),
        desc="Evaluating optimal linkage method and metric",
        total=total_combinations,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        with suppress(Exception):
            Z = linkage(m, method=method, metric=metric)
            threshold, score = _find_best_silhouette_score(Z, m, metric, linkage_criterion)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_threshold = threshold
                best_overall_method = method
                best_overall_metric = metric

    return best_overall_method, best_overall_metric, best_overall_threshold


def _find_best_silhouette_score(
    Z: np.ndarray,
    m: np.ndarray,
    linkage_metric: str,
    linkage_criterion: str,
    lower_bound: float = 0.001,
    upper_bound: float = 1.0,
    resolution: float = 0.001,
) -> Tuple[float, float]:
    """Find the best silhouette score using binary search.

    Args:
        Z (np.ndarray): Linkage matrix.
        m (np.ndarray): Data matrix.
        linkage_metric (str): Linkage metric for silhouette score calculation.
        linkage_criterion (str): Clustering criterion.
        lower_bound (float, optional): Lower bound for search. Defaults to 0.001.
        upper_bound (float, optional): Upper bound for search. Defaults to 1.0.
        resolution (float, optional): Desired resolution for the best threshold. Defaults to 0.001.

    Returns:
        Tuple[float, float]:
            - Best threshold (float): The threshold that yields the best silhouette score.
            - Best silhouette score (float): The highest silhouette score achieved.
    """
    best_score = -np.inf
    best_threshold = None

    # Test lower bound
    max_d_lower = np.max(Z[:, 2]) * lower_bound
    clusters_lower = fcluster(Z, max_d_lower, criterion=linkage_criterion)
    try:
        score_lower = silhouette_score(m, clusters_lower, metric=linkage_metric)
    except ValueError:
        score_lower = -np.inf

    # Test upper bound
    max_d_upper = np.max(Z[:, 2]) * upper_bound
    clusters_upper = fcluster(Z, max_d_upper, criterion=linkage_criterion)
    try:
        score_upper = silhouette_score(m, clusters_upper, metric=linkage_metric)
    except ValueError:
        score_upper = -np.inf

    # Determine initial bounds for binary search
    if score_lower > score_upper:
        best_score = score_lower
        best_threshold = lower_bound
        upper_bound = (lower_bound + upper_bound) / 2
    else:
        best_score = score_upper
        best_threshold = upper_bound
        lower_bound = (lower_bound + upper_bound) / 2

    # Binary search loop
    while upper_bound - lower_bound > resolution:
        mid_threshold = (upper_bound + lower_bound) / 2
        max_d_mid = np.max(Z[:, 2]) * mid_threshold
        clusters_mid = fcluster(Z, max_d_mid, criterion=linkage_criterion)
        try:
            score_mid = silhouette_score(m, clusters_mid, metric=linkage_metric)
        except ValueError:
            score_mid = -np.inf

        # Update best score and threshold if mid-point is better
        if score_mid > best_score:
            best_score = score_mid
            best_threshold = mid_threshold

        # Adjust bounds based on the scores
        if score_lower > score_upper:
            upper_bound = mid_threshold
        else:
            lower_bound = mid_threshold

    return best_threshold, float(best_score)
