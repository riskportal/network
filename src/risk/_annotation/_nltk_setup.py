"""
risk/_annotation/_nltk_setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import os
import zipfile
from typing import List, Optional, Tuple

import nltk
from nltk.data import find
from nltk.data import path as nltk_data_path

from .._log import logger


def setup_nltk_resources(required_resources: Optional[List[Tuple[str, str]]] = None) -> None:
    """
    Ensures all required NLTK resources are available and properly extracted.
    Uses NLTK's default paths and mechanisms.

    Args:
        required_resources (List[Tuple[str, str]], optional): List of required resources
            to download and extract. Each tuple should contain the resource path within
            NLTK data and the package name. Defaults to None.
    """
    if required_resources is None:
        required_resources = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
        ]

    # Process each resource
    for resource_path, package_name in required_resources:
        try:
            # First try to find the resource - this is how NLTK checks if it's available
            find(resource_path)
        except LookupError:
            # Resource not found, download it
            logger.info(f"Downloading missing NLTK resource: {package_name}")
            nltk.download(package_name, quiet=True)

        # Even if find() succeeded, the resource might be a zip that failed to extract
        # Check if we need to manually extract zips
        verify_and_extract_if_needed(resource_path, package_name)


def verify_and_extract_if_needed(resource_path: str, package_name: str) -> None:
    """
    Verifies if the resource is properly extracted and extracts if needed. Respects
    NLTK's directory structure where the extracted content should be in the same directory
    as the zip file.

    Args:
        resource_path (str): Path to the resource within NLTK data.
        package_name (str): Name of the NLTK package.
    """
    # Get the directory and base name from the resource path
    path_parts = resource_path.split("/")
    resource_type = path_parts[0]  # 'corpora', 'tokenizers', etc.
    resource_name = path_parts[-1]  # 'wordnet', 'punkt', etc.

    # Check all NLTK data directories
    for base in nltk_data_path:
        # For resource paths like 'corpora/wordnet', the zip file is at '~/nltk_data/corpora/wordnet.zip'
        # and the extracted directory should be at '~/nltk_data/corpora/wordnet'
        resource_dir = os.path.join(base, resource_type)
        zip_path = os.path.join(resource_dir, f"{resource_name}.zip")
        folder_path = os.path.join(resource_dir, resource_name)

        # If zip exists but folder doesn't, extraction is needed
        if os.path.exists(zip_path) and not os.path.exists(folder_path):
            logger.info(f"Found unextracted zip for {package_name}, extracting...")
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    # Extract files to the same directory where the zip file is located
                    zf.extractall(path=resource_dir)

                if os.path.exists(folder_path):
                    logger.info(f"Successfully extracted {package_name}")
                else:
                    logger.warning(
                        f"Extraction completed but resource directory not found for {package_name}"
                    )
            except Exception as e:
                logger.error(f"Failed to extract {package_name}: {e}")
