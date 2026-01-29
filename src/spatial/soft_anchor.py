# src/spatial/soft_anchor.py

"""
Soft spatial anchoring module for sports analytics system.

Performs gentle stabilization of field-space positions using weak geometric cues
from pitch line detection. This is NOT camera calibration or homography estimation.

What soft anchoring does:
- Applies small, gradual corrections to reduce jitter in field-space coordinates
- Uses pitch line orientation as a weak hint for rotation alignment
- Blends corrections conservatively with original positions
- Smooths correction parameters over time to avoid jumps

What soft anchoring does NOT do:
- Does not snap positions or teleport entities
- Does not compute camera calibration or homography
- Does not assume specific pitch dimensions or orientation
- Does not identify specific pitch features (sideline, goal line, etc.)
- Does not override base projection - only suggests gentle adjustments

Failure philosophy:
- When uncertain, return inputs unchanged
- When cues are weak or inconsistent, decay corrections to zero
- Never accumulate irreversible corrections
- All adjustments are temporary and reversible

This module improves visual stability, not geometric accuracy.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from collections import deque


class SoftAnchor:
    """
    Soft spatial anchoring for field-space position stabilization.
    
    Uses weak geometric cues from pitch lines to apply conservative corrections
    that reduce jitter without compromising the underlying projection.
    
    State is minimal and decay-oriented:
    - Stores recent orientation hints for temporal smoothing
    - Decays corrections when cues disappear
    - Never accumulates permanent bias
    
    Usage:
        anchor = SoftAnchor()
        anchored_pos = anchor.stabilize(raw_pos, line_cues, frame_index)
    """
    
    # Conservative thresholds
    MIN_CUE_CONFIDENCE = 0.6  # Only use strong cues
    MAX_ROTATION_CORRECTION = np.radians(5)  # Maximum 5° rotation per frame
    MAX_POSITION_SHIFT = 0.02  # Maximum 2% image shift
    ANCHOR_WEIGHT = 0.15  # Blend weight for anchored positions (very conservative)
    
    # Temporal smoothing
    SMOOTHING_WINDOW = 10  # Frames to average orientation hints
    CORRECTION_DECAY = 0.95  # Decay rate when cues are absent
    
    def __init__(self):
        """Initialize soft anchor with minimal state."""
        # Temporal smoothing for orientation hint
        self._orientation_history: deque = deque(maxlen=self.SMOOTHING_WINDOW)
        
        # Current correction state (decays when cues absent)
        self._rotation_correction: float = 0.0  # Radians
        self._scale_hint: float = 1.0  # Relative scale adjustment
        
        # Frame tracking for temporal smoothing
        self._last_frame: Optional[int] = None
    
    def stabilize(
        self,
        position: Tuple[float, float],
        line_cues: Optional[List[Any]] = None,
        frame_index: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Apply soft anchoring to a single field-space position.
        
        Returns gently stabilized position or original position if uncertain.
        
        Args:
            position: Raw field-space position (x, y) in normalized [0, 1] coords
            line_cues: Optional list of LineCue objects from field_lines.py
            frame_index: Optional frame index for temporal smoothing
        
        Returns:
            Anchored position (x, y) in same coordinate space as input.
            Returns input unchanged if anchoring is not confident.
        """
        # Fallback: no cues provided
        if line_cues is None or len(line_cues) == 0:
            self._decay_corrections()
            return position
        
        # Update correction parameters from cues
        correction_applied = self._update_corrections(line_cues, frame_index)
        
        # Fallback: cues too weak or inconsistent
        if not correction_applied:
            self._decay_corrections()
            return position
        
        # Apply conservative correction
        anchored = self._apply_correction(position)
        
        # Blend with original position (very conservative weight)
        x_final = (1 - self.ANCHOR_WEIGHT) * position[0] + self.ANCHOR_WEIGHT * anchored[0]
        y_final = (1 - self.ANCHOR_WEIGHT) * position[1] + self.ANCHOR_WEIGHT * anchored[1]
        
        # Safety check: reject if shift is too large
        shift = np.sqrt((x_final - position[0])**2 + (y_final - position[1])**2)
        if shift > self.MAX_POSITION_SHIFT:
            return position
        
        return (x_final, y_final)
    
    def stabilize_batch(
        self,
        positions: List[Tuple[float, float]],
        line_cues: Optional[List[Any]] = None,
        frame_index: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        """
        Apply soft anchoring to multiple positions at once.
        
        All positions share the same correction parameters for consistency.
        
        Args:
            positions: List of raw field-space positions (x, y)
            line_cues: Optional list of LineCue objects from field_lines.py
            frame_index: Optional frame index for temporal smoothing
        
        Returns:
            List of anchored positions in same order as input.
            Returns inputs unchanged if anchoring is not confident.
        """
        if len(positions) == 0:
            return positions
        
        # Update corrections once for all positions
        if line_cues is None or len(line_cues) == 0:
            self._decay_corrections()
            return positions
        
        correction_applied = self._update_corrections(line_cues, frame_index)
        if not correction_applied:
            self._decay_corrections()
            return positions
        
        # Apply to all positions
        anchored_positions = []
        for pos in positions:
            anchored = self._apply_correction(pos)
            
            # Blend with original
            x_final = (1 - self.ANCHOR_WEIGHT) * pos[0] + self.ANCHOR_WEIGHT * anchored[0]
            y_final = (1 - self.ANCHOR_WEIGHT) * pos[1] + self.ANCHOR_WEIGHT * anchored[1]
            
            # Safety check
            shift = np.sqrt((x_final - pos[0])**2 + (y_final - pos[1])**2)
            if shift > self.MAX_POSITION_SHIFT:
                anchored_positions.append(pos)
            else:
                anchored_positions.append((x_final, y_final))
        
        return anchored_positions
    
    def _update_corrections(
        self,
        line_cues: List[Any],
        frame_index: Optional[int]
    ) -> bool:
        """
        Update correction parameters from line cues.
        
        Args:
            line_cues: List of LineCue objects
            frame_index: Optional frame index
        
        Returns:
            True if corrections were updated, False if cues too weak
        """
        try:
            # Select only the strongest cue
            strongest_cue = max(line_cues, key=lambda c: c.confidence)
            
            # Reject if confidence too low
            if strongest_cue.confidence < self.MIN_CUE_CONFIDENCE:
                return False
            
            # Extract orientation hint
            orientation = strongest_cue.orientation
            
            # Add to temporal smoothing window
            self._orientation_history.append(orientation)
            
            # Compute smoothed orientation
            if len(self._orientation_history) > 0:
                # Use circular mean for angles
                angles = np.array(list(self._orientation_history))
                mean_cos = np.mean(np.cos(angles))
                mean_sin = np.mean(np.sin(angles))
                smoothed_orientation = np.arctan2(mean_sin, mean_cos)
            else:
                smoothed_orientation = orientation
            
            # Compute rotation correction (assume we want to align to horizontal/vertical)
            # Use the closest cardinal direction as target
            cardinal_angles = [0, np.pi/2, -np.pi/2]  # Horizontal, vertical up, vertical down
            closest_cardinal = min(cardinal_angles, key=lambda a: abs(a - smoothed_orientation))
            
            # Rotation correction is difference from cardinal
            rotation_delta = closest_cardinal - smoothed_orientation
            
            # Clamp rotation correction
            rotation_delta = np.clip(rotation_delta, -self.MAX_ROTATION_CORRECTION, self.MAX_ROTATION_CORRECTION)
            
            # Smooth update to current correction (gradual change)
            self._rotation_correction = 0.7 * self._rotation_correction + 0.3 * rotation_delta
            
            # Scale hint based on line length (longer lines suggest better calibration)
            # This is a very weak hint - mostly identity
            length_hint = strongest_cue.length
            target_scale = 1.0 + 0.02 * (length_hint - 0.2)  # Very small scale adjustment
            target_scale = np.clip(target_scale, 0.98, 1.02)
            
            # Smooth scale update
            self._scale_hint = 0.8 * self._scale_hint + 0.2 * target_scale
            
            self._last_frame = frame_index
            return True
        
        except Exception:
            # Silent failure
            return False
    
    def _apply_correction(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """
        Apply current correction parameters to a position.
        
        Args:
            position: Raw position (x, y)
        
        Returns:
            Corrected position (x, y)
        """
        x, y = position
        
        # Center around origin for rotation
        x_centered = x - 0.5
        y_centered = y - 0.5
        
        # Apply rotation correction
        cos_r = np.cos(self._rotation_correction)
        sin_r = np.sin(self._rotation_correction)
        x_rotated = x_centered * cos_r - y_centered * sin_r
        y_rotated = x_centered * sin_r + y_centered * cos_r
        
        # Apply scale hint
        x_scaled = x_rotated * self._scale_hint
        y_scaled = y_rotated * self._scale_hint
        
        # Translate back
        x_final = x_scaled + 0.5
        y_final = y_scaled + 0.5
        
        # Clamp to valid range
        x_final = np.clip(x_final, 0.0, 1.0)
        y_final = np.clip(y_final, 0.0, 1.0)
        
        return (x_final, y_final)
    
    def _decay_corrections(self) -> None:
        """
        Decay correction parameters when cues are absent or weak.
        
        This ensures corrections don't persist when evidence disappears.
        """
        # Decay rotation correction toward zero
        self._rotation_correction *= self.CORRECTION_DECAY
        
        # Decay scale hint toward identity (1.0)
        self._scale_hint = self._scale_hint * self.CORRECTION_DECAY + 1.0 * (1 - self.CORRECTION_DECAY)
        
        # Clear orientation history gradually
        if len(self._orientation_history) > 0:
            # Remove oldest entry to gradually forget
            if len(self._orientation_history) == self._orientation_history.maxlen:
                # Pop from left side (oldest)
                self._orientation_history.popleft()
    
    def reset(self) -> None:
        """
        Reset all correction state.
        
        Use when starting a new video sequence or when spatial context changes.
        """
        self._orientation_history.clear()
        self._rotation_correction = 0.0
        self._scale_hint = 1.0
        self._last_frame = None
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current anchor state for debugging or visualization.
        
        Returns:
            Dictionary with current correction parameters
        """
        return {
            'rotation_correction_deg': np.degrees(self._rotation_correction),
            'scale_hint': self._scale_hint,
            'orientation_history_size': len(self._orientation_history),
            'last_frame': self._last_frame
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<SoftAnchor "
            f"rot={np.degrees(self._rotation_correction):.2f}° "
            f"scale={self._scale_hint:.3f} "
            f"history={len(self._orientation_history)}>"
        )