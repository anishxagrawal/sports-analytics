# src/analytics/smoothing.py

"""
Trajectory smoothing utilities for sports analytics system.

Provides pure functions for smoothing noisy position sequences.
Does not modify input data - returns new smoothed sequences.
"""

from typing import List, Tuple, Sequence


def smooth_trajectory_ema(
    positions: Sequence[Tuple[float, float]],
    alpha: float = 0.3
) -> List[Tuple[float, float]]:
    """
    Smooth a trajectory using Exponential Moving Average.
    
    EMA applies more weight to recent positions while incorporating history.
    Formula: smoothed[i] = alpha * raw[i] + (1 - alpha) * smoothed[i-1]
    
    Args:
        positions: Sequence of (x, y) positions to smooth
        alpha: Smoothing factor (0 < alpha <= 1)
               - Higher alpha = less smoothing (follows raw data closely)
               - Lower alpha = more smoothing (smoother but more lag)
    
    Returns:
        New list of smoothed (x, y) positions with same length as input
    
    Notes:
        - Input positions are never modified
        - First position is kept unchanged (no history to smooth against)
        - Empty input returns empty list
        - alpha clamped to valid range [0.01, 1.0]
    """
    if not positions:
        return []
    
    # Clamp alpha to valid range
    alpha = max(0.01, min(1.0, alpha))
    
    smoothed = []
    
    # First position remains unchanged (no previous smoothed value)
    smoothed.append(positions[0])
    
    # Apply EMA to subsequent positions
    for i in range(1, len(positions)):
        x_raw, y_raw = positions[i]
        x_prev, y_prev = smoothed[i - 1]
        
        # EMA: new = alpha * raw + (1 - alpha) * previous_smoothed
        x_smooth = alpha * x_raw + (1.0 - alpha) * x_prev
        y_smooth = alpha * y_raw + (1.0 - alpha) * y_prev
        
        smoothed.append((x_smooth, y_smooth))
    
    return smoothed