# src/visualization/pitch_overlay.py
"""
2D pitch overlay visualization for debug purposes.

Draws a miniature top-down pitch view in the corner of the video frame,
showing player and ball positions in normalized field space.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class PitchOverlay:
    """
    Renders a 2D top-down pitch overlay with entity positions.
    
    The pitch uses normalized field coordinates [0,1] x [0,1] where:
    - (0, 0) is bottom-left corner of the field
    - (1, 1) is top-right corner of the field

    """
    
    def __init__(
        self,
        width: int = 300,
        height: int = 200,
        position: str = "top-right",
        margin: int = 20,
        alpha: float = 0.85,
        enabled: bool = True
    ):
        """
        Initialize the pitch overlay renderer.
        
        Args:
            width: Width of the overlay in pixels
            height: Height of the overlay in pixels
            position: Corner placement ("top-left", "top-right", "bottom-left", "bottom-right")
            margin: Margin from video frame edges in pixels
            alpha: Opacity of the overlay (0.0 = transparent, 1.0 = opaque)
            enabled: Whether visualization is active
        """
        self.width = width
        self.height = height
        self.position = position
        self.margin = margin
        self.alpha = alpha
        self.enabled = enabled
        
        # Pitch drawing margins (space around the field inside the overlay)
        self.pitch_margin = 15
        
        # Calculate actual drawable pitch area
        self.pitch_width = width - 2 * self.pitch_margin
        self.pitch_height = height - 2 * self.pitch_margin
        
        # Colors (BGR format for OpenCV)
        self.color_background = (34, 139, 34)  # Forest green
        self.color_lines = (255, 255, 255)      # White
        self.color_player = (0, 255, 255)       # Yellow
        self.color_ball = (0, 140, 255)         # Orange
        self.color_border = (200, 200, 200)     # Light gray
        
        # Cache the base pitch (static, computed once)
        self.base_pitch = self._create_base_pitch()
    
    def _create_base_pitch(self) -> np.ndarray:
        """
        Create a static pitch image with field markings.
        
        Returns:
            Base pitch image as BGR numpy array
        """
        # Create blank overlay with green background
        overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        overlay[:] = self.color_background
        
        # Define pitch boundaries in overlay coordinates
        x_min = self.pitch_margin
        y_min = self.pitch_margin
        x_max = self.pitch_margin + self.pitch_width
        y_max = self.pitch_margin + self.pitch_height
        
        # Draw outer boundary
        cv2.rectangle(
            overlay,
            (x_min, y_min),
            (x_max, y_max),
            self.color_lines,
            2
        )
        
        # Draw halfway line (vertical center line)
        halfway_x = x_min + self.pitch_width // 2
        cv2.line(
            overlay,
            (halfway_x, y_min),
            (halfway_x, y_max),
            self.color_lines,
            1
        )
        
        # Draw center circle
        center_x = x_min + self.pitch_width // 2
        center_y = y_min + self.pitch_height // 2
        radius = int(self.pitch_height * 0.15)  # ~15% of pitch height
        cv2.circle(
            overlay,
            (center_x, center_y),
            radius,
            self.color_lines,
            1
        )
        
        # Draw center spot
        cv2.circle(
            overlay,
            (center_x, center_y),
            2,
            self.color_lines,
            -1
        )
        
        # Draw border around entire overlay
        cv2.rectangle(
            overlay,
            (0, 0),
            (self.width - 1, self.height - 1),
            self.color_border,
            1
        )
        
        return overlay
    
    def _field_to_overlay(self, field_x: float, field_y: float) -> Tuple[int, int]:
        """
        Convert normalized field coordinates to overlay pixel coordinates.
        
        Args:
            field_x: Normalized X coordinate [0, 1]
            field_y: Normalized Y coordinate [0, 1]
            
        Returns:
            (pixel_x, pixel_y) in overlay coordinate space
        """
        # Clamp to valid range
        field_x = np.clip(field_x, 0.0, 1.0)
        field_y = np.clip(field_y, 0.0, 1.0)
        
        # Flip Y axis: field (0,0) is bottom-left, overlay (0,0) is top-left
        field_y = 1.0 - field_y
        
        # Map to overlay coordinates
        pixel_x = int(self.pitch_margin + field_x * self.pitch_width)
        pixel_y = int(self.pitch_margin + field_y * self.pitch_height)
        
        return pixel_x, pixel_y
    
    def _calculate_overlay_position(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int]:
        """
        Calculate top-left corner position of overlay on the video frame.
        
        Args:
            frame_shape: Shape of the video frame (height, width, channels)
            
        Returns:
            (y, x) coordinates for top-left corner of overlay
        """
        frame_height, frame_width = frame_shape[:2]
        
        if self.position == "top-left":
            y = self.margin
            x = self.margin
        elif self.position == "top-right":
            y = self.margin
            x = frame_width - self.width - self.margin
        elif self.position == "bottom-left":
            y = frame_height - self.height - self.margin
            x = self.margin
        elif self.position == "bottom-right":
            y = frame_height - self.height - self.margin
            x = frame_width - self.width - self.margin
        else:
            # Default to top-right
            y = self.margin
            x = frame_width - self.width - self.margin
        
        return y, x
    
    def render(
        self,
        frame: np.ndarray,
        players: List[Dict],
        ball: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Render the pitch overlay onto a video frame.
        
        Args:
            frame: Video frame as BGR numpy array
            players: List of player dicts with 'field_position' or 'field_position_anchored'
            ball: Optional ball dict with 'field_position' or 'field_position_anchored'
            
        Returns:
            Frame with overlay blended in
        """
        # Early exit if disabled
        if not self.enabled:
            return frame
        
        # Copy the cached base pitch
        overlay = self.base_pitch.copy()
        
        # Draw players
        for player in players:
            # Try anchored position first, fall back to regular field position
            field_pos = player.get('field_position_anchored') or player.get('field_position')
            
            if field_pos is not None and len(field_pos) >= 2:
                field_x, field_y = field_pos[0], field_pos[1]
                pixel_x, pixel_y = self._field_to_overlay(field_x, field_y)
                
                # Draw player as small filled circle
                cv2.circle(
                    overlay,
                    (pixel_x, pixel_y),
                    4,
                    self.color_player,
                    -1
                )
                
                # Optional: draw outline for visibility
                cv2.circle(
                    overlay,
                    (pixel_x, pixel_y),
                    4,
                    (0, 0, 0),
                    1
                )
        
        # Draw ball (larger and distinct)
        if ball is not None:
            field_pos = ball.get('field_position_anchored') or ball.get('field_position')
            
            if field_pos is not None and len(field_pos) >= 2:
                field_x, field_y = field_pos[0], field_pos[1]
                pixel_x, pixel_y = self._field_to_overlay(field_x, field_y)
                
                # Draw ball as larger filled circle
                cv2.circle(
                    overlay,
                    (pixel_x, pixel_y),
                    6,
                    self.color_ball,
                    -1
                )
                
                # Draw outline for visibility
                cv2.circle(
                    overlay,
                    (pixel_x, pixel_y),
                    6,
                    (0, 0, 0),
                    1
                )
        
        # Blend overlay onto frame
        y_pos, x_pos = self._calculate_overlay_position(frame.shape)
        
        # Extract region of interest from frame
        roi = frame[y_pos:y_pos + self.height, x_pos:x_pos + self.width]
        
        # Defensive check: ensure ROI matches overlay dimensions
        if roi.shape[:2] != overlay.shape[:2]:
            return frame
        
        # Alpha blend overlay onto ROI
        blended = cv2.addWeighted(
            overlay,
            self.alpha,
            roi,
            1.0 - self.alpha,
            0
        )
        
        # Place blended overlay back into frame
        frame[y_pos:y_pos + self.height, x_pos:x_pos + self.width] = blended
        
        return frame


# === Integration Example ===
# 
# In main.py, after entity updates and before writing frame:
#
# # Initialize once before frame loop:
# pitch_overlay = PitchOverlay(
#     width=300,
#     height=200,
#     position="top-right",
#     margin=20,
#     alpha=0.85,
#     enabled=True  # Toggle visualization on/off
# )
#
# # Inside frame loop, after entity_manager updates:
# # Collect player positions
# players_for_viz = []
# for player in entity_manager.get_active_players():
#     if hasattr(player, 'field_position') or hasattr(player, 'field_position_anchored'):
#         players_for_viz.append({
#             'field_position': getattr(player, 'field_position', None),
#             'field_position_anchored': getattr(player, 'field_position_anchored', None)
#         })
#
# # Get ball position
# ball_for_viz = None
# if entity_manager.ball.is_visible():
#     ball_for_viz = {
#         'field_position': getattr(entity_manager.ball, 'field_position', None),
#         'field_position_anchored': getattr(entity_manager.ball, 'field_position_anchored', None)
#     }
#
# # Render overlay
# frame = pitch_overlay.render(frame, players_for_viz, ball_for_viz)
#
# # Then continue with writer.write(frame)
# ===========================