"""
risk/_network/_graph/_summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

from ..._log import log_header, logger


class Summary:
    """
    Handles the processing, storage, and export of network analysis results.

    The Results class provides methods to process significance and depletion data, compute
    FDR-corrected q-values, and structure information on domains and annotations into a
    DataFrame. It also offers functionality to export the processed data in CSV, JSON,
    and text formats for analysis and reporting.
    """

    def __init__(
        self,
        annotation: Dict[str, Any],
        neighborhoods: Dict[str, Any],
        graph,  # Avoid type hinting Graph to prevent circular imports
    ):
        """
        Initialize the Results object with analysis components.

        Args:
            annotation (Dict[str, Any]): Annotation data, including ordered annotations and matrix of associations.
            neighborhoods (Dict[str, Any]): Neighborhood data containing p-values for significance and depletion analysis.
            graph (Graph): Graph object representing domain-to-node and node-to-label mappings.
        """
        self.annotation = annotation
        self.neighborhoods = neighborhoods
        self.graph = graph

    def to_csv(self, filepath: str) -> None:
        """
        Export significance results to a CSV file.

        Args:
            filepath (str): The path where the CSV file will be saved.
        """
        # Load results and export directly to CSV
        results = self.load()
        results.to_csv(filepath, index=False)
        logger.info(f"Analysis summary exported to CSV file: {filepath}")

    def to_json(self, filepath: str) -> None:
        """
        Export significance results to a JSON file.

        Args:
            filepath (str): The path where the JSON file will be saved.
        """
        # Load results and export directly to JSON
        results = self.load()
        results.to_json(filepath, orient="records", indent=4)
        logger.info(f"Analysis summary exported to JSON file: {filepath}")

    def to_txt(self, filepath: str) -> None:
        """
        Export significance results to a text file.

        Args:
            filepath (str): The path where the text file will be saved.
        """
        # Load results and export directly to text file
        results = self.load()
        with open(filepath, "w", encoding="utf-8") as txt_file:
            txt_file.write(results.to_string(index=False))

        logger.info(f"Analysis summary exported to text file: {filepath}")

    def load(self) -> pd.DataFrame:
        """
        Load and process domain and annotation data into a DataFrame with significance metrics.

        Returns:
            pd.DataFrame: Processed DataFrame containing significance scores, p-values, q-values,
                and annotation member information.
        """
        log_header("Loading analysis summary")
        # Calculate significance and depletion q-values from p-value matrices in annotation
        enrichment_pvals = self.neighborhoods["enrichment_pvals"]
        depletion_pvals = self.neighborhoods["depletion_pvals"]
        enrichment_qvals = self._calculate_qvalues(enrichment_pvals)
        depletion_qvals = self._calculate_qvalues(depletion_pvals)

        # Initialize DataFrame with domain and annotation details
        results = pd.DataFrame(
            [
                {"Domain ID": domain_id, "Annotation": desc, "Summed Significance Score": score}
                for domain_id, info in self.graph.domain_id_to_domain_info_map.items()
                for desc, score in zip(info["full_descriptions"], info["significance_scores"])
            ]
        )
        # Sort by Domain ID and Summed Significance Score
        results = results.sort_values(
            by=["Domain ID", "Summed Significance Score"], ascending=[True, False]
        ).reset_index(drop=True)

        # Add minimum p-values and q-values to DataFrame
        results[
            [
                "Enrichment P-Value",
                "Enrichment Q-value",
                "Depletion P-Value",
                "Depletion Q-value",
            ]
        ] = results.apply(
            lambda row: self._get_significance_values(
                row["Domain ID"],
                row["Annotation"],
                enrichment_pvals,
                depletion_pvals,
                enrichment_qvals,
                depletion_qvals,
            ),
            axis=1,
            result_type="expand",
        )
        # Add annotation members and their counts
        results["Annotation Members in Network"] = results["Annotation"].apply(
            lambda desc: self._get_annotation_members(desc)
        )
        results["Annotation Members in Network Count"] = results[
            "Annotation Members in Network"
        ].apply(lambda x: len(x.split(";")) if x else 0)

        # Reorder columns and drop rows with NaN values
        results = (
            results[
                [
                    "Domain ID",
                    "Annotation",
                    "Annotation Members in Network",
                    "Annotation Members in Network Count",
                    "Summed Significance Score",
                    "Enrichment P-Value",
                    "Enrichment Q-value",
                    "Depletion P-Value",
                    "Depletion Q-value",
                ]
            ]
            .dropna()
            .reset_index(drop=True)
        )

        # Convert annotation list to a DataFrame for comparison then merge with results
        ordered_annotation = pd.DataFrame({"Annotation": self.annotation["ordered_annotation"]})
        # Merge to ensure all annotations are present, filling missing rows with defaults
        results = pd.merge(ordered_annotation, results, on="Annotation", how="left").fillna(
            {
                "Domain ID": -1,
                "Annotation Members in Network": "",
                "Annotation Members in Network Count": 0,
                "Summed Significance Score": 0.0,
                "Enrichment P-Value": 1.0,
                "Enrichment Q-value": 1.0,
                "Depletion P-Value": 1.0,
                "Depletion Q-value": 1.0,
            }
        )
        # Convert "Domain ID" and "Annotation Members in Network Count" to integers
        results["Domain ID"] = results["Domain ID"].astype(int)
        results["Annotation Members in Network Count"] = results[
            "Annotation Members in Network Count"
        ].astype(int)

        return results

    def _calculate_qvalues(self, pvals: np.ndarray) -> np.ndarray:
        """
        Calculate q-values (FDR) for each row of a p-value matrix.

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
        enrichment_pvals: np.ndarray,
        depletion_pvals: np.ndarray,
        enrichment_qvals: np.ndarray,
        depletion_qvals: np.ndarray,
    ) -> Tuple[Union[float, None], Union[float, None], Union[float, None], Union[float, None]]:
        """
        Retrieve the most significant p-values and q-values (FDR) for a given annotation.

        Args:
            domain_id (int): The domain ID associated with the annotation.
            description (str): The annotation description.
            enrichment_pvals (np.ndarray): Matrix of significance p-values.
            depletion_pvals (np.ndarray): Matrix of depletion p-values.
            enrichment_qvals (np.ndarray): Matrix of significance q-values.
            depletion_qvals (np.ndarray): Matrix of depletion q-values.

        Returns:
            Tuple[Union[float, None], Union[float, None], Union[float, None], Union[float, None]]:
                Minimum significance p-value, significance q-value, depletion p-value, depletion q-value.
        """
        try:
            annotation_idx = self.annotation["ordered_annotation"].index(description)
        except ValueError:
            return None, None, None, None  # Description not found

        node_indices = self.graph.domain_id_to_node_ids_map.get(domain_id, [])
        if not node_indices:
            return None, None, None, None  # No associated nodes

        sig_p = enrichment_pvals[node_indices, annotation_idx]
        dep_p = depletion_pvals[node_indices, annotation_idx]
        sig_q = enrichment_qvals[node_indices, annotation_idx]
        dep_q = depletion_qvals[node_indices, annotation_idx]

        return (
            np.min(sig_p) if sig_p.size > 0 else None,
            np.min(sig_q) if sig_q.size > 0 else None,
            np.min(dep_p) if dep_p.size > 0 else None,
            np.min(dep_q) if dep_q.size > 0 else None,
        )

    def _get_annotation_members(self, description: str) -> str:
        """
        Retrieve node labels associated with a given annotation description.

        Args:
            description (str): The annotation description.

        Returns:
            str: ';'-separated string of node labels that are associated with the annotation.
        """
        try:
            annotation_idx = self.annotation["ordered_annotation"].index(description)
        except ValueError:
            return ""  # Description not found

        # Get the column (safely) from the sparse matrix
        column = self.annotation["matrix"][:, annotation_idx]
        # Convert the column to a dense array if needed
        column = column.toarray().ravel()  # Convert to a 1D dense array
        # Get nodes present for the annotation and sort by node label - use np.where on the dense array
        nodes_present = np.where(column == 1)[0]
        node_labels = sorted(
            self.graph.node_id_to_node_label_map[node_id]
            for node_id in nodes_present
            if node_id in self.graph.node_id_to_node_label_map
        )
        return ";".join(node_labels)
