# src/visualization/object_renderer.py

"""
Object visualization renderer for sports analytics.

Provides stateless functions to draw tracked objects with motion metrics
(speed, direction, distance traveled) on video frames.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Any


def draw_object_with_metrics(
    frame: np.ndarray,
    position: Tuple[float, float],
    track_id: int,
    speed_px_per_sec: float,
    direction: Tuple[float, float],
    distance_traveled: float,
    class_id: int,
    circle_radius: int = 15
) -> None:
    """
    Draw circle with directional arrow and motion metrics for a tracked object.
    
    Renders on frame in-place. Shows circle, velocity arrow, and text labels
    for speed and distance traveled. Optimized for smooth, clean visualization.
    
    Args:
        frame: Video frame (modified in-place)
        position: (x, y) center position in pixels
        track_id: Object tracker ID for label
        speed_px_per_sec: Speed magnitude in pixels per second
        direction: (dx, dy) normalized direction vector (should be unit vector)
        distance_traveled: Total distance traveled in pixels
        class_id: Object class (0=Player, 1=Ball, 2=Referee)
        circle_radius: Radius of circle in pixels
    
    Notes:
        - Modifies frame in-place
        - Arrow length scales with speed for visibility
        - Uses anti-aliasing for smooth rendering
        - Stateless function; creates no side effects
    """
    if position is None or direction is None:
        return
    
    x, y = int(position[0]), int(position[1])
    dx, dy = direction
    
    # Color by class (BGR format)
    if class_id == 1:
        color = (0, 0, 255)  # Ball: red
    elif class_id == 0:
        color = (255, 0, 0)  # Player: blue
    elif class_id == 2:
        color = (0, 255, 255)  # Referee: yellow
    else:
        color = (255, 255, 255)  # Unknown: white
    
    # Draw outer circle (thicker, smoother)
    cv2.circle(frame, (x, y), circle_radius, color, 3, lineType=cv2.LINE_AA)
    
    # Draw inner center dot
    cv2.circle(frame, (x, y), 2, color, -1, lineType=cv2.LINE_AA)
    
    # Draw directional arrow (scale arrow length by speed for visual feedback)
    # Clamp arrow length between 20 and 60 pixels
    arrow_length = max(20, min(int(speed_px_per_sec / 5), 60))
    arrow_end_x = int(x + dx * arrow_length)
    arrow_end_y = int(y + dy * arrow_length)
    
    cv2.arrowedLine(
        frame,
        (x, y),
        (arrow_end_x, arrow_end_y),
        color,
        2,
        tipLength=0.25,
        line_type=cv2.LINE_AA
    )
    
    # Draw speed metric (semi-transparent background for readability)
    speed_text = f"ID:{track_id} {speed_px_per_sec:.0f}px/s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        speed_text, font, font_scale, thickness
    )
    
    # Draw semi-transparent background
    text_x = x - 40
    text_y = y + circle_radius + 15
    cv2.rectangle(
        frame,
        (text_x - 2, text_y - text_height - 2),
        (text_x + text_width + 2, text_y + baseline + 2),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        speed_text,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA
    )
    
    # Draw distance metric
    distance_text = f"d:{distance_traveled:.0f}px"
    (dist_width, dist_height), dist_baseline = cv2.getTextSize(
        distance_text, font, font_scale, thickness
    )
    
    dist_x = x - 40
    dist_y = y + circle_radius + 30
    cv2.rectangle(
        frame,
        (dist_x - 2, dist_y - dist_height - 2),
        (dist_x + dist_width + 2, dist_y + dist_baseline + 2),
        (0, 0, 0),
        -1
    )
    
    cv2.putText(
        frame,
        distance_text,
        (dist_x, dist_y),
        font,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA
    )


def draw_players_with_metrics(
    frame: np.ndarray,
    players: list,
    fps: float,
    circle_radius: int = 15
) -> None:
    """
    Draw all active players with motion metrics on frame.
    
    Convenience function that iterates over players and renders each one.
    
    Args:
        frame: Video frame (modified in-place)
        players: List of player entities with position/trajectory methods
        fps: Video frame rate (used to compute speed)
        circle_radius: Radius of circle in pixels
    
    Notes:
        - Imports compute_speed, compute_direction, compute_distance_traveled
          lazily to avoid circular imports
        - Modifies frame in-place
    """
    from analytics.motion import compute_speed, compute_direction, compute_distance_traveled
    
    for player in players:
        position = player.get_position()
        if position is None:
            continue
        
        speed = compute_speed(player, fps)
        direction = compute_direction(player, normalize=True)
        distance = compute_distance_traveled(player)
        
        draw_object_with_metrics(
            frame,
            position=position,
            track_id=player.track_id,
            speed_px_per_sec=speed,
            direction=direction if direction else (0, 0),
            distance_traveled=distance,
            class_id=player.class_id,
            circle_radius=circle_radius
        )


def draw_ball_with_metrics(
    frame: np.ndarray,
    ball: Any,
    fps: float,
    circle_radius: int = 8
) -> None:
    """
    Draw ball with motion metrics on frame.
    
    Args:
        frame: Video frame (modified in-place)
        ball: Ball entity with position/trajectory methods
        fps: Video frame rate (used to compute speed)
        circle_radius: Radius of circle in pixels
    
    Notes:
        - Imports motion functions lazily to avoid circular imports
        - Modifies frame in-place
    """
    from analytics.motion import compute_distance_traveled
    
    if not ball.is_visible():
        return
    
    ball_position = ball.get_position()
    if ball_position is None:
        return
    
    # Compute ball motion metrics
    ball_trajectory = ball.get_trajectory()
    if len(ball_trajectory) >= 2:
        last_pos = ball_trajectory[-1]
        prev_pos = ball_trajectory[-2]
        ball_dx = last_pos[0] - prev_pos[0]
        ball_dy = last_pos[1] - prev_pos[1]
        mag = np.sqrt(ball_dx**2 + ball_dy**2)
        if mag > 1e-6:
            ball_direction = (ball_dx / mag, ball_dy / mag)
        else:
            ball_direction = (0, 0)
    else:
        ball_direction = (0, 0)
    
    # Compute speed from last two positions
    ball_speed = 0.0
    if len(ball_trajectory) >= 2:
        last_pos = ball_trajectory[-1]
        prev_pos = ball_trajectory[-2]
        distance = np.sqrt((last_pos[0] - prev_pos[0])**2 + (last_pos[1] - prev_pos[1])**2)
        ball_speed = distance * fps
    
    ball_distance = compute_distance_traveled(ball)
    
    draw_object_with_metrics(
        frame,
        position=ball_position,
        track_id=-1,  # Ball doesn't have a meaningful track ID
        speed_px_per_sec=ball_speed,
        direction=ball_direction,
        distance_traveled=ball_distance,
        class_id=1,  # Ball class
        circle_radius=circle_radius
    )
