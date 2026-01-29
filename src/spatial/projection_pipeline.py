# src/spatial/projection_pipeline.py

"""
Spatial projection pipeline for sports analytics system.

Orchestrates the conversion of detected objects from image space to stable
field-space positions. This is the coordination layer that connects spatial
components in the correct order.

Pipeline stages:
1. Extract ground contact points from bounding boxes (image space)
2. Convert ground points to normalized field space [0, 1] × [0, 1]
3. Apply optional soft spatial anchoring for stability
4. Return stable field-space positions

This module is glue code, not a solver. It delegates all math and vision logic
to specialized components and focuses on correct sequencing and safe failure
handling.

Philosophy:
- Consistency over accuracy
- Fail safely by skipping objects rather than propagating invalid data
- No silent failures - invalid data is discarded, not approximated
- Minimal state - only what's needed for coordination
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from spatial.ground_point import bbox_to_ground_point
from spatial.image_to_field import pixel_to_field
from spatial.soft_anchor import SoftAnchor
from spatial.field_lines import detect_field_lines


class ProjectionPipeline:
    """
    Orchestrates spatial projection from image space to field space.
    
    Connects ground point extraction, field normalization, and soft anchoring
    in a safe, sequential pipeline. Handles failures gracefully by skipping
    invalid objects.
    
    Usage:
        pipeline = ProjectionPipeline(enable_anchoring=True)
        
        for frame in video:
            line_cues = detect_field_lines(frame)
            results = pipeline.process_frame(
                detections=detections,
                frame_shape=(height, width),
                line_cues=line_cues,
                frame_index=i
            )
    """
    
    def __init__(
        self,
        enable_anchoring: bool = True,
        enable_line_detection: bool = False
    ):
        """
        Initialize projection pipeline.
        
        Args:
            enable_anchoring: Whether to apply soft spatial anchoring
            enable_line_detection: Whether to detect field lines automatically
                (if False, line_cues must be provided to process_frame)
        """
        self.enable_anchoring = enable_anchoring
        self.enable_line_detection = enable_line_detection
        
        # Initialize soft anchor if enabled
        self.soft_anchor = SoftAnchor() if enable_anchoring else None
    
    def process_frame(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        frame: Optional[np.ndarray] = None,
        line_cues: Optional[List[Any]] = None,
        frame_index: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all detections in a single frame through the projection pipeline.
        
        Pipeline stages:
        1. Extract ground points from bounding boxes
        2. Project to normalized field space
        3. Apply soft anchoring (optional)
        4. Return enriched detection dictionaries
        
        Args:
            detections: List of detection dictionaries, each with:
                - 'bbox': tuple (x1, y1, x2, y2)
                - 'track_id': int (optional)
                - 'class_id': int (optional)
                - 'confidence': float (optional)
            frame_shape: Frame dimensions (height, width)
            frame: Optional frame for automatic line detection
            line_cues: Optional pre-computed line cues from detect_field_lines()
            frame_index: Optional frame index for temporal smoothing
        
        Returns:
            List of detection dictionaries enriched with:
            - 'ground_point': tuple (x, y) in image space
            - 'field_position': tuple (fx, fy) in [0, 1] × [0, 1]
            - 'field_position_anchored': tuple (fx, fy) after soft anchoring (if enabled)
            
            Objects that fail any projection stage are omitted from results.
        """
        # Detect field lines if enabled and frame provided
        if self.enable_line_detection and frame is not None and line_cues is None:
            try:
                line_cues = detect_field_lines(frame)
            except Exception:
                # Silent failure - proceed without line cues
                line_cues = None
        
        results = []
        
        for detection in detections:
            try:
                # Stage 1: Extract ground contact point from bounding box
                ground_point = self._extract_ground_point(detection)
                if ground_point is None:
                    continue  # Skip this detection
                
                # Stage 2: Project ground point to field space
                field_position = self._project_to_field(ground_point, frame_shape)
                if field_position is None:
                    continue  # Skip this detection
                
                # Stage 3: Apply soft anchoring (optional)
                if self.enable_anchoring and self.soft_anchor is not None:
                    field_position_anchored = self._apply_anchoring(
                        field_position,
                        line_cues,
                        frame_index
                    )
                else:
                    field_position_anchored = field_position
                
                # Enrich detection with spatial data
                enriched = detection.copy()
                enriched['ground_point'] = ground_point
                enriched['field_position'] = field_position
                enriched['field_position_anchored'] = field_position_anchored
                
                results.append(enriched)
            
            except Exception:
                # Silent failure - skip this detection and continue
                continue
        
        return results
    
    def process_positions(
        self,
        positions: List[Tuple[float, float]],
        frame_shape: Tuple[int, int],
        frame: Optional[np.ndarray] = None,
        line_cues: Optional[List[Any]] = None,
        frame_index: Optional[int] = None
    ) -> List[Optional[Tuple[float, float]]]:
        """
        Process a list of ground points (already in image space) to field space.
        
        Useful when ground points are already computed elsewhere (e.g., from
        entity tracking system).
        
        Args:
            positions: List of ground points (x, y) in image space
            frame_shape: Frame dimensions (height, width)
            frame: Optional frame for automatic line detection
            line_cues: Optional pre-computed line cues
            frame_index: Optional frame index for temporal smoothing
        
        Returns:
            List of field positions (fx, fy) or None for failed projections.
            Length matches input list.
        """
        # Detect field lines if enabled and frame provided
        if self.enable_line_detection and frame is not None and line_cues is None:
            try:
                line_cues = detect_field_lines(frame)
            except Exception:
                line_cues = None
        
        results = []
        
        for position in positions:
            try:
                # Stage 1: Project to field space
                field_position = self._project_to_field(position, frame_shape)
                if field_position is None:
                    results.append(None)
                    continue
                
                # Stage 2: Apply soft anchoring (optional)
                if self.enable_anchoring and self.soft_anchor is not None:
                    field_position_anchored = self._apply_anchoring(
                        field_position,
                        line_cues,
                        frame_index
                    )
                else:
                    field_position_anchored = field_position
                
                results.append(field_position_anchored)
            
            except Exception:
                # Silent failure
                results.append(None)
        
        return results
    
    def _extract_ground_point(
        self,
        detection: Dict[str, Any]
    ) -> Optional[Tuple[float, float]]:
        """
        Extract ground contact point from detection bounding box.
        
        Stage 1 of pipeline: bbox → ground point (image space)
        
        Args:
            detection: Detection dictionary with 'bbox' key
        
        Returns:
            Ground point (x, y) in image space or None on failure
        """
        try:
            bbox = detection.get('bbox')
            if bbox is None:
                return None
            
            ground_point = bbox_to_ground_point(bbox)
            return ground_point
        
        except Exception:
            return None
    
    def _project_to_field(
        self,
        ground_point: Tuple[float, float],
        frame_shape: Tuple[int, int]
    ) -> Optional[Tuple[float, float]]:
        """
        Project ground point from image space to normalized field space.
        
        Stage 2 of pipeline: ground point → field position [0, 1] × [0, 1]
        
        Args:
            ground_point: Ground point (x, y) in image space
            frame_shape: Frame dimensions (height, width)
        
        Returns:
            Field position (fx, fy) in [0, 1] × [0, 1] or None on failure
        """
        try:
            field_position = pixel_to_field(ground_point, frame_shape)
            return field_position
        
        except Exception:
            return None
    
    def _apply_anchoring(
        self,
        field_position: Tuple[float, float],
        line_cues: Optional[List[Any]],
        frame_index: Optional[int]
    ) -> Tuple[float, float]:
        """
        Apply soft spatial anchoring for stability.
        
        Stage 3 of pipeline: field position → anchored field position
        
        If anchoring fails or is uncertain, returns original position unchanged.
        
        Args:
            field_position: Raw field position (fx, fy)
            line_cues: Optional line cues from field line detection
            frame_index: Optional frame index for temporal smoothing
        
        Returns:
            Anchored field position (fx, fy), or original position if anchoring fails
        """
        try:
            if self.soft_anchor is None:
                return field_position
            
            anchored = self.soft_anchor.stabilize(
                position=field_position,
                line_cues=line_cues,
                frame_index=frame_index
            )
            
            return anchored
        
        except Exception:
            # Fallback to original position on any error
            return field_position
    
    def reset(self) -> None:
        """
        Reset pipeline state.
        
        Useful when starting a new video sequence or when spatial context
        changes significantly (e.g., camera cut, halftime).
        """
        if self.soft_anchor is not None:
            self.soft_anchor.reset()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current pipeline state for debugging.
        
        Returns:
            Dictionary with pipeline configuration and component states
        """
        state = {
            'enable_anchoring': self.enable_anchoring,
            'enable_line_detection': self.enable_line_detection
        }
        
        if self.soft_anchor is not None:
            state['soft_anchor'] = self.soft_anchor.get_state()
        
        return state
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<ProjectionPipeline "
            f"anchoring={self.enable_anchoring} "
            f"line_detection={self.enable_line_detection}>"
        )