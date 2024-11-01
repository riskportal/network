"""
risk/log/results
~~~~~~~~~~~~~~~~
"""

import warnings
from functools import lru_cache
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

from risk.network.graph import NetworkGraph

# Suppress all warnings - this is to resolve warnings from multiprocessing
warnings.filterwarnings("ignore")


class Results:
    """Handles the processing, storage, and export of network analysis results.

    The Results class provides methods to process significance and depletion data, compute
    FDR-corrected q-values, and structure information on domains and annotations into a
    DataFrame. It also offers functionality to export the processed data in CSV, JSON,
    and text formats for analysis and reporting.
    """

    def __init__(
        self, annotations: Dict[str, Any], neighborhoods: Dict[str, Any], graph: NetworkGraph
    ):
        """Initialize the Results object with analysis components.

        Args:
            annotations (Dict[str, Any]): Annotation data, including ordered annotations and matrix of associations.
            neighborhoods (Dict[str, Any]): Neighborhood data containing p-values for significance and depletion analysis.
            graph (NetworkGraph): Graph object representing domain-to-node and node-to-label mappings.
        """
        self.annotations = annotations
        self.neighborhoods = neighborhoods
        self.graph = graph

    def to_csv(self, filepath: str) -> None:
        """Export significance results to a CSV file.

        Args:
            filepath (str): The path where the CSV file will be saved.
        """
        # Load results and export directly to CSV
        results = self.load()
        results.to_csv(filepath, index=False)

    def to_json(self, filepath: str) -> None:
        """Export significance results to a JSON file.

        Args:
            filepath (str): The path where the JSON file will be saved.
        """
        # Load results and export directly to JSON
        results = self.load()
        results.to_json(filepath, orient="records", indent=4)

    def to_txt(self, filepath: str) -> None:
        """Export significance results to a text file.

        Args:
            filepath (str): The path where the text file will be saved.
        """
        # Load results and export directly to text file
        results = self.load()
        with open(filepath, "w") as txt_file:
            txt_file.write(results.to_string(index=False))

    @lru_cache(maxsize=None)
    def load(self) -> pd.DataFrame:
        """Load and process domain and annotation data into a DataFrame with significance metrics.

        Args:
            graph (Any): Graph object containing domain-to-node and node-to-label mappings.
            annotations (Dict[str, Any]): Annotation details, including ordered annotations and matrix.

        Returns:
            pd.DataFrame: Processed DataFrame containing significance scores, p-values, q-values,
                and annotation member information.
        """
        # Calculate significance and depletion q-values from p-value matrices in `annotations`
        sig_pvals = self.neighborhoods["significance_pvals"]
        dep_pvals = self.neighborhoods["depletion_pvals"]
        sig_qvals = self._calculate_qvalues(sig_pvals)
        dep_qvals = self._calculate_qvalues(dep_pvals)

        # Initialize DataFrame with domain and annotation details
        results = pd.DataFrame(
            [
                {"Domain ID": domain_id, "Annotation": desc, "Summed Significance Score": score}
                for domain_id, info in self.graph.domain_id_to_domain_info_map.items()
                for desc, score in zip(info["full_descriptions"], info["enrichment_scores"])
            ]
        )
        # Sort by Domain ID and Summed Significance Score
        results = results.sort_values(
            by=["Domain ID", "Summed Significance Score"], ascending=[True, False]
        ).reset_index(drop=True)

        # Add minimum p-values and q-values to DataFrame
        results[
            [
                "Significance P-value",
                "Significance Q-value",
                "Depletion P-value",
                "Depletion Q-value",
            ]
        ] = results.apply(
            lambda row: self._get_significance_values(
                self.annotations,
                self.graph,
                row["Domain ID"],
                row["Annotation"],
                sig_pvals,
                dep_pvals,
                sig_qvals,
                dep_qvals,
            ),
            axis=1,
            result_type="expand",
        )
        # Add annotation members and their counts
        results["Annotation Members"] = results["Annotation"].apply(
            lambda desc: self._get_annotation_members(desc, self.annotations, self.graph)
        )
        results["Annotation Member Count"] = results["Annotation Members"].apply(
            lambda x: len(x.split(";")) if x else 0
        )

        # Reorder columns and drop rows with NaN values
        results = (
            results[
                [
                    "Domain ID",
                    "Annotation",
                    "Annotation Members",
                    "Annotation Member Count",
                    "Summed Significance Score",
                    "Significance P-value",
                    "Significance Q-value",
                    "Depletion P-value",
                    "Depletion Q-value",
                ]
            ]
            .dropna()
            .reset_index(drop=True)
        )

        return results

    @staticmethod
    def _calculate_qvalues(pvals: np.ndarray) -> np.ndarray:
        """Calculate q-values (FDR) for each row of a p-value matrix.

        Args:
            pvals (np.ndarray): 2D array of p-values.

        Returns:
            np.ndarray: 2D array of q-values, with FDR correction applied row-wise.
        """
        return np.apply_along_axis(lambda row: fdrcorrection(row)[1], 1, pvals)

    def _get_significance_values(
        self,
        domain_id: int,
        description: str,
        sig_pvals: np.ndarray,
        dep_pvals: np.ndarray,
        sig_qvals: np.ndarray,
        dep_qvals: np.ndarray,
    ) -> Tuple[Union[float, None], Union[float, None], Union[float, None], Union[float, None]]:
        """Retrieve the most significant p-values and q-values (FDR) for a given annotation.

        Args:
            domain_id (int): The domain ID associated with the annotation.
            description (str): The annotation description.
            sig_pvals (np.ndarray): Matrix of significance p-values.
            dep_pvals (np.ndarray): Matrix of depletion p-values.
            sig_qvals (np.ndarray): Matrix of significance q-values.
            dep_qvals (np.ndarray): Matrix of depletion q-values.

        Returns:
            Tuple[Union[float, None], Union[float, None], Union[float, None], Union[float, None]]:
                Minimum significance p-value, significance q-value, depletion p-value, depletion q-value.
        """
        try:
            annotation_idx = self.annotations["ordered_annotations"].index(description)
        except ValueError:
            return None, None, None, None  # Description not found

        node_indices = self.graph.domain_id_to_node_ids_map.get(domain_id, [])
        if not node_indices:
            return None, None, None, None  # No associated nodes

        sig_p = sig_pvals[node_indices, annotation_idx]
        dep_p = dep_pvals[node_indices, annotation_idx]
        sig_q = sig_qvals[node_indices, annotation_idx]
        dep_q = dep_qvals[node_indices, annotation_idx]

        return (
            np.min(sig_p) if sig_p.size > 0 else None,
            np.min(sig_q) if sig_q.size > 0 else None,
            np.min(dep_p) if dep_p.size > 0 else None,
            np.min(dep_q) if dep_q.size > 0 else None,
        )

    def _get_annotation_members(self, description: str) -> str:
        """Retrieve node labels associated with a given annotation description.

        Args:
            description (str): The annotation description.

        Returns:
            str: ';'-separated string of node labels that are associated with the annotation.
        """
        try:
            annotation_idx = self.annotations["ordered_annotations"].index(description)
        except ValueError:
            return ""  # Description not found

        nodes_present = np.where(self.annotations["matrix"][:, annotation_idx] == 1)[0]
        node_labels = sorted(
            self.graph.node_id_to_node_label_map[node_id]
            for node_id in nodes_present
            if node_id in self.graph.node_id_to_node_label_map
        )
        return ";".join(node_labels)
