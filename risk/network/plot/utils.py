"""
risk/network/plot/utils
~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import List, Tuple, Union

import matplotlib.colors as mcolors
import numpy as np

from risk.network.graph import NetworkGraph


def get_annotated_domain_colors(
    graph: NetworkGraph,
    cmap: str = "gist_rainbow",
    color: Union[str, None] = None,
    min_scale: float = 0.8,
    max_scale: float = 1.0,
    scale_factor: float = 1.0,
    random_seed: int = 888,
) -> np.ndarray:
    """Get colors for the domains based on node annotations, or use a specified color.

    Args:
        graph (NetworkGraph): The network data and attributes to be visualized.
        cmap (str, optional): Colormap to use for generating domain colors. Defaults to "gist_rainbow".
        color (str or None, optional): Color to use for the domains. If None, the colormap will be used. Defaults to None.
        min_scale (float, optional): Minimum scale for color intensity when generating domain colors.
            Defaults to 0.8.
        max_scale (float, optional): Maximum scale for color intensity when generating domain colors.
            Defaults to 1.0.
        scale_factor (float, optional): Factor for adjusting the contrast in the colors generated based on
            enrichment. Higher values increase the contrast. Defaults to 1.0.
        random_seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 888.

    Returns:
        np.ndarray: Array of RGBA colors for each domain.
    """
    # Generate domain colors based on the enrichment data
    node_colors = graph.get_domain_colors(
        cmap=cmap,
        color=color,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_factor=scale_factor,
        random_seed=random_seed,
    )
    annotated_colors = []
    for _, node_ids in graph.domain_id_to_node_ids_map.items():
        if len(node_ids) > 1:
            # For multi-node domains, choose the brightest color based on RGB sum
            domain_colors = np.array([node_colors[node] for node in node_ids])
            brightest_color = domain_colors[
                np.argmax(domain_colors[:, :3].sum(axis=1))  # Sum the RGB values
            ]
            annotated_colors.append(brightest_color)
        else:
            # Single-node domains default to white (RGBA)
            default_color = np.array([1.0, 1.0, 1.0, 1.0])
            annotated_colors.append(default_color)

    return np.array(annotated_colors)


def calculate_bounding_box(
    node_coordinates: np.ndarray, radius_margin: float = 1.05
) -> Tuple[np.ndarray, float]:
    """Calculate the bounding box of the network based on node coordinates.

    Args:
        node_coordinates (np.ndarray): Array of node coordinates (x, y).
        radius_margin (float, optional): Margin factor to apply to the bounding box radius. Defaults to 1.05.

    Returns:
        tuple: Center of the bounding box and the radius (adjusted by the radius margin).
    """
    # Find minimum and maximum x, y coordinates
    x_min, y_min = np.min(node_coordinates, axis=0)
    x_max, y_max = np.max(node_coordinates, axis=0)
    # Calculate the center of the bounding box
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    # Calculate the radius of the bounding box, adjusted by the margin
    radius = max(x_max - x_min, y_max - y_min) / 2 * radius_margin
    return center, radius


def to_rgba(
    color: Union[str, List, Tuple, np.ndarray],
    alpha: Union[float, None] = None,
    num_repeats: Union[int, None] = None,
) -> np.ndarray:
    """Convert color(s) to RGBA format, applying alpha and repeating as needed.

    Args:
        color (Union[str, list, tuple, np.ndarray]): The color(s) to convert. Can be a string, list, tuple, or np.ndarray.
        alpha (float, None, optional): Alpha value (transparency) to apply. If provided, it overrides any existing alpha values
            found in color.
        num_repeats (int, None, optional): If provided, the color(s) will be repeated this many times. Defaults to None.

    Returns:
        np.ndarray: Array of RGBA colors repeated `num_repeats` times, if applicable.
    """

    def convert_to_rgba(c: Union[str, List, Tuple, np.ndarray]) -> np.ndarray:
        """Convert a single color to RGBA format, handling strings, hex, and RGB/RGBA lists."""
        # Note: if no alpha is provided, the default alpha value is 1.0 by mcolors.to_rgba
        if isinstance(c, str):
            # Convert color names or hex values (e.g., 'red', '#FF5733') to RGBA
            rgba = np.array(mcolors.to_rgba(c))
        elif isinstance(c, (list, tuple, np.ndarray)) and len(c) in [3, 4]:
            # Convert RGB (3) or RGBA (4) values to RGBA format
            rgba = np.array(mcolors.to_rgba(c))
        else:
            raise ValueError(
                f"Invalid color format: {c}. Must be a valid string or RGB/RGBA sequence."
            )

        if alpha is not None:  # Override alpha if provided
            rgba[3] = alpha
        return rgba

    # If color is a 2D array of RGBA values, convert it to a list of lists
    if isinstance(color, np.ndarray) and color.ndim == 2 and color.shape[1] == 4:
        color = [list(c) for c in color]

    # Handle a single color (string or RGB/RGBA list/tuple)
    if isinstance(color, (str, list, tuple)) and not any(
        isinstance(c, (list, tuple, np.ndarray)) for c in color
    ):
        rgba_color = convert_to_rgba(color)
        if num_repeats:
            return np.tile(
                rgba_color, (num_repeats, 1)
            )  # Repeat the color if num_repeats is provided
        return np.array([rgba_color])  # Return a single color wrapped in a numpy array

    # Handle a list/array of colors
    elif isinstance(color, (list, tuple, np.ndarray)):
        rgba_colors = np.array(
            [convert_to_rgba(c) for c in color]
        )  # Convert each color in the list to RGBA
        # Handle repetition if num_repeats is provided
        if num_repeats:
            repeated_colors = np.array(
                [rgba_colors[i % len(rgba_colors)] for i in range(num_repeats)]
            )
            return repeated_colors

        return rgba_colors

    else:
        raise ValueError("Color must be a valid RGB/RGBA or array of RGB/RGBA colors.")
