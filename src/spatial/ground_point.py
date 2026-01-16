# src/spatial/ground_point.py
"""
Spatial interpretation utility for computing ground contact points from bounding boxes.

This module provides a pure, stateless function to convert screen-space bounding boxes
into ground contact points, assuming the bottom-center of the box represents where
the subject makes contact with the ground plane.
"""

from typing import Tuple


def bbox_to_ground_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Convert a bounding box to a ground contact point.
    
    Computes the bottom-center point of the bounding box, which represents
    the estimated ground contact location (e.g., feet position for a person).
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2) where:
            - x1: Left edge (pixels)
            - y1: Top edge (pixels)
            - x2: Right edge (pixels)
            - y2: Bottom edge (pixels)
    
    Returns:
        Tuple of (x, y) coordinates representing the ground contact point:
        - x: Horizontal center of the box
        - y: Bottom edge of the box (ground contact)
    """
    x1, y1, x2, y2 = bbox
    
    ground_x = (x1 + x2) / 2.0
    ground_y = y2
    
    return (ground_x, ground_y)