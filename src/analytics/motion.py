# src/analytics/motion.py

"""
Motion analytics module for sports analytics system.

Provides stateless functions for computing movement metrics from Player
entity position histories. All functions are pure and do not modify entities.
"""

from typing import Optional, Tuple
import numpy as np
from core.smoothing import smooth_trajectory_ema


def compute_speed(player, fps: float, use_smoothing: bool = False) -> float:
    """
    Compute current speed of a player.
    
    Uses the last two positions from player trajectory to estimate
    instantaneous speed in pixels per second.
    
    Args:
        player: Player entity with get_trajectory() method
        fps: Video frame rate (frames per second)
        use_smoothing: If True, apply EMA smoothing to ground positions
    
    Returns:
        Speed in pixels per second, or 0.0 if insufficient history
    
    Notes:
        - Requires at least 2 positions in trajectory
        - Speed is Euclidean distance / time_delta
        - Time delta is computed as 1/fps (one frame interval)
    """
    if fps <= 0:
        return 0.0
    
    if hasattr(player, 'ground_positions') and len(player.ground_positions) >= 2:
        positions = list(player.ground_positions)
        
        if use_smoothing:
            positions = smooth_trajectory_ema(positions, alpha=0.3)
        
        pos_prev = positions[-2]
        pos_curr = positions[-1]
    else:
        trajectory = player.get_trajectory(n=2)
        
        if len(trajectory) < 2:
            return 0.0
        
        pos_prev = trajectory[-2]
        pos_curr = trajectory[-1]
    
    dx = pos_curr[0] - pos_prev[0]
    dy = pos_curr[1] - pos_prev[1]
    
    distance = np.sqrt(dx**2 + dy**2)
    time_delta = 1.0 / fps
    
    speed = distance / time_delta
    
    return float(speed)


def compute_direction(
    player,
    normalize: bool = True,
    use_smoothing: bool = False
) -> Optional[Tuple[float, float]]:
    """
    Compute current movement direction of a player.
    
    Uses the last two positions from player trajectory to compute
    direction vector (dx, dy).
    
    Args:
        player: Player entity with get_trajectory() method
        normalize: If True, return unit vector; if False, return raw displacement
        use_smoothing: If True, apply EMA smoothing to ground positions
    
    Returns:
        Tuple of (dx, dy) as direction vector, or None if insufficient history
    
    Notes:
        - Requires at least 2 positions in trajectory
        - If normalize=True and player is stationary, returns (0.0, 0.0)
        - Direction points from previous position to current position
    """
    if hasattr(player, 'ground_positions') and len(player.ground_positions) >= 2:
        positions = list(player.ground_positions)
        
        if use_smoothing:
            positions = smooth_trajectory_ema(positions, alpha=0.3)
        
        pos_prev = positions[-2]
        pos_curr = positions[-1]
    else:
        trajectory = player.get_trajectory(n=2)
        
        if len(trajectory) < 2:
            return None
        
        pos_prev = trajectory[-2]
        pos_curr = trajectory[-1]
    
    dx = pos_curr[0] - pos_prev[0]
    dy = pos_curr[1] - pos_prev[1]
    
    if not normalize:
        return (float(dx), float(dy))
    
    magnitude = np.sqrt(dx**2 + dy**2)
    
    if magnitude < 1e-6:
        return (0.0, 0.0)
    
    dx_norm = dx / magnitude
    dy_norm = dy / magnitude
    
    return (float(dx_norm), float(dy_norm))


def compute_field_velocity(player, fps: float, anchored: bool = True) -> Tuple[float, float]:
    """
    Compute current velocity of a player in field space.
    
    Uses the last two field-space positions to estimate instantaneous
    velocity in field-units per second.
    
    Args:
        player: Player entity with field position attributes
        fps: Video frame rate (frames per second)
        anchored: If True, prefer field_positions_anchored; otherwise use field_positions
    
    Returns:
        Tuple of (vx, vy) in field-units per second, or (0.0, 0.0) if insufficient data
    
    Notes:
        - Requires at least 2 positions in the selected field position history
        - Field space is normalized [0,1] x [0,1]
        - Velocity is computed as displacement / time_delta
        - Time delta is computed as 1/fps (one frame interval)
    """
    if fps <= 0:
        return (0.0, 0.0)
    
    positions = None
    
    if anchored:
        if hasattr(player, 'field_positions_anchored') and player.field_positions_anchored:
            positions = list(player.field_positions_anchored)
        elif hasattr(player, 'field_positions') and player.field_positions:
            positions = list(player.field_positions)
    else:
        if hasattr(player, 'field_positions') and player.field_positions:
            positions = list(player.field_positions)
    
    if positions is None or len(positions) < 2:
        return (0.0, 0.0)
    
    pos_prev = positions[-2]
    pos_curr = positions[-1]
    
    if pos_prev is None or pos_curr is None:
        return (0.0, 0.0)
    
    dx = pos_curr[0] - pos_prev[0]
    dy = pos_curr[1] - pos_prev[1]
    
    time_delta = 1.0 / fps
    
    vx = dx / time_delta
    vy = dy / time_delta
    
    return (float(vx), float(vy))


def compute_field_speed(player, fps: float, anchored: bool = True) -> float:
    """
    Compute current speed of a player in field space.
    
    Computes scalar speed from field velocity magnitude.
    
    Args:
        player: Player entity with field position attributes
        fps: Video frame rate (frames per second)
        anchored: If True, prefer field_position_anchored; otherwise use field_position
    
    Returns:
        Speed in field-units per second, or 0.0 if insufficient data
    
    Notes:
        - Speed is the magnitude of the velocity vector
        - Uses compute_field_velocity internally
    """
    vx, vy = compute_field_velocity(player, fps, anchored)
    
    speed = np.sqrt(vx**2 + vy**2)
    
    return float(speed)


def compute_distance_traveled(entity, max_history: Optional[int] = None) -> float:
    """
    Compute total distance traveled by an entity.
    
    Sums Euclidean distances between consecutive positions in the entity's trajectory.
    
    Args:
        entity: Entity with get_trajectory() method
        max_history: Optional limit on how many recent positions to consider.
                    If None, uses entire trajectory.
    
    Returns:
        Total distance traveled in pixels, or 0.0 if insufficient history
    
    Notes:
        - Requires at least 2 positions in trajectory
        - Distance is computed in pixel space (not field space)
        - Stateless function; does not modify entity
    """
    trajectory = entity.get_trajectory()
    
    if len(trajectory) < 2:
        return 0.0
    
    # Optionally limit to recent history
    if max_history is not None and max_history > 0:
        trajectory = trajectory[-max_history:]
    
    total_distance = 0.0
    for i in range(len(trajectory) - 1):
        pt1 = trajectory[i]
        pt2 = trajectory[i + 1]
        distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        total_distance += distance
    
    return float(total_distance)