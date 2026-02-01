# src/spatial/world_projection_pipeline.py

"""
Extended projection pipeline with world coordinate support.

Adds homography-based camera motion compensation to the base projection pipeline.

Pipeline stages:
1. Detect pitch lines and extract keypoints
2. Estimate homography for current frame
3. Stabilize homography temporally
4. Project image → world coordinates (meters)
5. Compute real-world velocity and speed

This module wraps the base ProjectionPipeline and adds world-space capabilities.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import logging

from spatial.pitch_detector import detect_pitch_lines, extract_keypoints_from_lines
from spatial.homography_estimator import estimate_homography_ransac
from spatial.homography_buffer import TemporalHomographyBuffer
from spatial.world_projector import WorldProjector
from spatial.world_velocity import WorldVelocityEstimator
from spatial.ground_point import bbox_to_ground_point

logger = logging.getLogger(__name__)


class WorldProjectionPipeline:
    """
    Complete pipeline from image detection to world-space metrics.
    
    Integrates:
    - Pitch line detection
    - Homography estimation with RANSAC
    - Temporal smoothing
    - World coordinate projection
    - Velocity estimation
    
    Usage:
        # Initialization (one-time)
        pipeline = WorldProjectionPipeline(
            reference_frame,
            reference_keypoints_world
        )
        
        # Per-frame
        for frame_idx, frame in enumerate(video):
            results = pipeline.process_frame(frame, detections, frame_idx)
            
            for result in results:
                print(f"Player at {result['world_position']} m/s "
                      f"with speed {result['speed']} m/s")
    """
    
    def __init__(
        self,
        reference_frame: np.ndarray,
        reference_keypoints_world: List[Tuple[float, float]],
        fps: float = 30.0,
        enable_homography: bool = True,
        enable_velocity: bool = True
    ):
        """
        Initialize world projection pipeline.
        
        Args:
            reference_frame: First frame with good pitch visibility
            reference_keypoints_world: List of (X, Y) in meters for reference points
            fps: Video frame rate
            enable_homography: Whether to use homography-based compensation
            enable_velocity: Whether to compute world velocities
        """
        
        self.fps = fps
        self.enable_homography = enable_homography
        self.enable_velocity = enable_velocity
        
        self.frame_count = 0
        
        # Step 1: Calibrate reference frame
        logger.info("Calibrating reference frame...")
        ref_lines = detect_pitch_lines(reference_frame)
        ref_keypoints_image = extract_keypoints_from_lines(
            ref_lines,
            reference_frame.shape[:2]
        )
        
        if len(ref_keypoints_image) < 4:
            logger.warning(
                f"Reference frame has only {len(ref_keypoints_image)} keypoints, "
                "expected >=4. World projection may be inaccurate."
            )
        
        logger.info(f"Found {len(ref_keypoints_image)} reference keypoints")
        
        # Step 2: Create world projector
        self.world_projector = WorldProjector(
            ref_keypoints_image,
            reference_keypoints_world,
            method='affine'
        )
        
        # Step 3: Initialize homography buffer
        self.homography_buffer = TemporalHomographyBuffer(
            window_size=5,
            smoothing_alpha=0.3
        )
        
        # Step 4: Store reference frame for homography matching
        self.reference_frame = reference_frame
        self.reference_lines = ref_lines
        self.reference_keypoints_image = ref_keypoints_image
        
        # Step 5: Velocity estimators (per track_id)
        self.velocity_estimators: Dict[int, WorldVelocityEstimator] = {}
        
        logger.info("WorldProjectionPipeline initialized successfully")
    
    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        frame_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Process frame through full world projection pipeline.
        
        Args:
            frame: Current frame (BGR)
            detections: List of detection dicts with 'bbox', 'class_id', 'track_id'
            frame_idx: Frame number
        
        Returns:
            Enriched detection dicts with world coordinates and velocities
        """
        
        self.frame_count += 1
        
        results = []
        
        # Stage 1: Estimate homography
        H_frame_to_ref, H_confidence, H_valid = self._estimate_frame_homography(
            frame, frame_idx
        )
        
        # Stage 2: Stabilize homography temporally
        H_stabilized = self.homography_buffer.add_observation(
            H_frame_to_ref if H_valid else None,
            H_confidence,
            frame_idx
        )
        
        # Stage 3: Project detections to world space
        for detection in detections:
            try:
                # Get bounding box
                bbox = detection.get('bbox')
                if bbox is None:
                    continue
                
                # Get ground point (bottom-center of bbox)
                ground_point = bbox_to_ground_point(bbox)
                if ground_point is None:
                    continue
                
                # Project to world coordinates
                world_position = self.world_projector.image_to_world(
                    ground_point,
                    H_stabilized if self.enable_homography else None
                )
                
                # Enrich detection
                enriched = detection.copy()
                enriched['ground_point'] = ground_point
                enriched['world_position'] = world_position
                enriched['homography_confidence'] = H_confidence
                
                # Stage 4: Compute velocity (if enabled)
                if self.enable_velocity:
                    track_id = detection.get('track_id', -1)
                    
                    if track_id not in self.velocity_estimators:
                        self.velocity_estimators[track_id] = WorldVelocityEstimator(
                            fps=self.fps,
                            entity_type='ball' if detection.get('class_id') == 1 else 'player'
                        )
                    
                    vx, vy, speed = self.velocity_estimators[track_id].update(
                        world_position,
                        frame_idx,
                        quality=H_confidence
                    )
                    
                    enriched['velocity'] = (vx, vy)
                    enriched['speed'] = speed
                
                results.append(enriched)
            
            except Exception as e:
                logger.debug(f"Error processing detection: {e}")
                continue
        
        return results
    
    def _estimate_frame_homography(
        self,
        frame: np.ndarray,
        frame_idx: int
    ) -> Tuple[Optional[np.ndarray], float, bool]:
        """
        Estimate homography from current frame to reference.
        
        Args:
            frame: Current frame
            frame_idx: Frame number
        
        Returns:
            (H, confidence, is_valid)
        """
        
        if not self.enable_homography:
            return np.eye(3), 1.0, True
        
        try:
            # Detect lines and keypoints in current frame
            lines_current = detect_pitch_lines(frame)
            keypoints_current_image = extract_keypoints_from_lines(
                lines_current,
                frame.shape[:2]
            )
            
            if len(keypoints_current_image) < 4:
                logger.debug(f"Frame {frame_idx}: Only {len(keypoints_current_image)} keypoints")
                return None, 0.0, False
            
            # Match keypoints to reference
            # Simple approach: use spatial proximity
            # More sophisticated: use descriptor matching
            keypoints_current_array = np.array(keypoints_current_image, dtype=np.float32)
            keypoints_ref_array = np.array(self.reference_keypoints_image, dtype=np.float32)
            
            # For simplicity: assume structural correspondence
            # In production: use feature matching or manual annotation
            if len(keypoints_current_array) >= 4 and len(keypoints_ref_array) >= 4:
                # Estimate homography
                result = estimate_homography_ransac(
                    keypoints_ref_array[:min(10, len(keypoints_ref_array))],
                    keypoints_current_array[:min(10, len(keypoints_current_array))],
                    max_reprojection_error=5.0
                )
                
                if result.is_valid:
                    return result.H, result.confidence, True
            
            return None, 0.0, False
        
        except Exception as e:
            logger.debug(f"Homography estimation failed: {e}")
            return None, 0.0, False
    
    def get_velocity_estimator(self, track_id: int) -> Optional[WorldVelocityEstimator]:
        """
        Get velocity estimator for a track.
        
        Args:
            track_id: Track ID
        
        Returns:
            WorldVelocityEstimator or None
        """
        
        return self.velocity_estimators.get(track_id)
    
    def reset(self) -> None:
        """
        Reset pipeline state.
        
        Call when starting new video or after significant event.
        """
        
        self.homography_buffer.reset()
        for est in self.velocity_estimators.values():
            est.reset()
        self.velocity_estimators.clear()
    
    def get_diagnostics(self) -> dict:
        """
        Get diagnostic information.
        
        Returns:
            Dictionary with pipeline stats
        """
        
        return {
            'frame_count': self.frame_count,
            'homography_buffer': self.homography_buffer.get_statistics(),
            'velocity_estimators': len(self.velocity_estimators),
            'enabled_features': {
                'homography': self.enable_homography,
                'velocity': self.enable_velocity
            }
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<WorldProjectionPipeline "
            f"frames={self.frame_count} "
            f"estimators={len(self.velocity_estimators)}>"
        )
