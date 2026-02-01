# src/spatial/world_projector.py

"""
World coordinate projection using calibrated homographies.

Projects image coordinates through homography chains to real-world
field coordinates (meters).

Pipeline:
1. Reference frame establishes image → world mapping
2. Per-frame homographies map current frame → reference frame
3. Compose transforms: image_t → image_ref → world
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PitchModel:
    """Standard football pitch dimensions and keypoints."""
    
    PITCH_LENGTH = 105.0  # meters
    PITCH_WIDTH = 68.0    # meters
    
    # Known structural keypoints on the pitch
    STRUCTURAL_KEYPOINTS = {
        'center': (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
        'left_goal': (0.0, PITCH_WIDTH / 2),
        'right_goal': (PITCH_LENGTH, PITCH_WIDTH / 2),
        'penalty_left': (11.0, PITCH_WIDTH / 2),
        'penalty_right': (PITCH_LENGTH - 11.0, PITCH_WIDTH / 2),
        'corner_tl': (0.0, PITCH_WIDTH),
        'corner_tr': (PITCH_LENGTH, PITCH_WIDTH),
        'corner_bl': (0.0, 0.0),
        'corner_br': (PITCH_LENGTH, 0.0),
    }
    
    @staticmethod
    def get_grid_keypoints(
        spacing_x: float = 5.0,
        spacing_y: float = 5.0
    ) -> dict:
        """
        Generate regular grid of known pitch intersections.
        
        Args:
            spacing_x: Spacing along pitch length (meters)
            spacing_y: Spacing along pitch width (meters)
        
        Returns:
            Dict mapping keypoint names to (X, Y) world coordinates
        """
        
        grid = {}
        
        for x in np.arange(0, PitchModel.PITCH_LENGTH + spacing_x, spacing_x):
            for y in np.arange(0, PitchModel.PITCH_WIDTH + spacing_y, spacing_y):
                key = f"grid_{x:.0f}_{y:.0f}"
                grid[key] = (x, y)
        
        return grid


class WorldProjector:
    """
    Projects image coordinates to world (field) coordinates.
    
    Maintains:
    - Reference frame calibration (image → world)
    - Per-frame homographies (current → reference image)
    - Composes transforms for projection
    
    Usage:
        projector = WorldProjector(ref_keypoints_image, ref_keypoints_world)
        
        # Each frame:
        H_frame_to_ref = estimate_homography(...)
        world_pos = projector.image_to_world(image_pos, H_frame_to_ref)
    """
    
    def __init__(
        self,
        reference_keypoints_image: List[Tuple[float, float]],
        reference_keypoints_world: List[Tuple[float, float]],
        method: str = 'affine'
    ):
        """
        Initialize projector with reference frame calibration.
        
        Args:
            reference_keypoints_image: List of (x, y) in reference image
            reference_keypoints_world: List of (X, Y) in world (meters)
            method: 'affine' (6 DOF) or 'homography' (8 DOF)
        """
        
        self.method = method
        self.reference_keypoints_image = reference_keypoints_image
        self.reference_keypoints_world = reference_keypoints_world
        
        # Fit mapping from reference image space to world space
        src_pts = np.array(reference_keypoints_image, dtype=np.float32)
        dst_pts = np.array(reference_keypoints_world, dtype=np.float32)
        
        if len(src_pts) < 3:
            raise ValueError("Need at least 3 reference keypoints")
        
        if method == 'affine':
            # Affine: 6 DOF (2×3 matrix)
            if len(src_pts) < 3:
                raise ValueError("Affine needs >=3 points")
            
            self.ref_transform = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
            self.ref_transform_inv = _affine_inverse(self.ref_transform)
        
        elif method == 'homography':
            # Homography: 8 DOF (3×3 matrix)
            if len(src_pts) < 4:
                raise ValueError("Homography needs >=4 points")
            
            self.ref_transform, _ = cv2.findHomography(src_pts[:4], dst_pts[:4])
            self.ref_transform_inv = np.linalg.inv(self.ref_transform)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(
            f"WorldProjector initialized with {len(src_pts)} reference points "
            f"using {method} method"
        )
    
    def image_to_world(
        self,
        image_point: Tuple[float, float],
        frame_homography: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Project image point in current frame to world coordinates.
        
        Pipeline:
        1. If frame_homography provided: transform image → reference image
        2. Transform reference image → world via calibration
        
        Args:
            image_point: (x, y) in current frame
            frame_homography: Optional 3×3 homography (current → reference)
        
        Returns:
            (X, Y) in world coordinates (meters)
        """
        
        x, y = image_point
        
        # Step 1: Transform to reference frame (if homography provided)
        if frame_homography is not None:
            pt_homog = np.array([x, y, 1.0])
            pt_ref_homog = frame_homography @ pt_homog
            pt_ref = pt_ref_homog[:2] / pt_ref_homog[2]
        else:
            pt_ref = np.array([x, y])
        
        # Step 2: Transform to world
        if self.method == 'affine':
            pt_homog = np.append(pt_ref, 1.0)
            world_pt = self.ref_transform @ pt_homog
        else:  # homography
            pt_homog = np.append(pt_ref, 1.0)
            world_homog = self.ref_transform @ pt_homog
            world_pt = world_homog[:2] / world_homog[2]
        
        return tuple(world_pt)
    
    def batch_image_to_world(
        self,
        image_points: np.ndarray,  # N×2
        frame_homography: Optional[np.ndarray] = None
    ) -> np.ndarray:  # N×2
        """
        Vectorized projection for multiple points.
        
        Args:
            image_points: N×2 array of (x, y)
            frame_homography: Optional 3×3 homography
        
        Returns:
            N×2 array of (X, Y) in world coordinates
        """
        
        image_points = np.asarray(image_points, dtype=np.float32)
        
        if image_points.ndim == 1:
            image_points = image_points.reshape(1, -1)
        
        N = image_points.shape[0]
        
        # Step 1: Transform to reference frame
        if frame_homography is not None:
            ones = np.ones((N, 1))
            pts_homog = np.hstack([image_points, ones])
            
            pts_ref_homog = (frame_homography @ pts_homog.T).T
            pts_ref = pts_ref_homog[:, :2] / pts_ref_homog[:, 2:3]
        else:
            pts_ref = image_points
        
        # Step 2: Transform to world
        if self.method == 'affine':
            ones = np.ones((N, 1))
            pts_homog = np.hstack([pts_ref, ones])
            world_pts = (self.ref_transform @ pts_homog.T).T
        else:  # homography
            ones = np.ones((N, 1))
            pts_homog = np.hstack([pts_ref, ones])
            world_homog = (self.ref_transform @ pts_homog.T).T
            world_pts = world_homog[:, :2] / world_homog[:, 2:3]
        
        return world_pts
    
    def world_to_image(
        self,
        world_point: Tuple[float, float],
        frame_homography: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Inverse projection: world → image (for validation).
        
        Args:
            world_point: (X, Y) in world coordinates
            frame_homography: Optional 3×3 homography (current ← reference)
        
        Returns:
            (x, y) in current frame
        """
        
        X, Y = world_point
        
        # Step 1: Transform to reference image
        if self.method == 'affine':
            pt_homog = np.append([X, Y], 1.0)
            pt_ref = self.ref_transform_inv @ pt_homog
        else:  # homography
            pt_homog = np.append([X, Y], 1.0)
            pt_ref_homog = self.ref_transform_inv @ pt_homog
            pt_ref = pt_ref_homog[:2] / pt_ref_homog[2]
        
        # Step 2: Transform to current frame (if homography provided)
        if frame_homography is not None:
            H_inv = np.linalg.inv(frame_homography)
            pt_homog = np.append(pt_ref, 1.0)
            pt_current_homog = H_inv @ pt_homog
            pt_current = pt_current_homog[:2] / pt_current_homog[2]
        else:
            pt_current = pt_ref
        
        return tuple(pt_current)
    
    def get_reference_calibration(self) -> dict:
        """
        Get reference calibration info for debugging.
        
        Returns:
            Dictionary with calibration details
        """
        
        return {
            'method': self.method,
            'num_reference_points': len(self.reference_keypoints_image),
            'reference_transform': self.ref_transform,
            'pitch_length': PitchModel.PITCH_LENGTH,
            'pitch_width': PitchModel.PITCH_WIDTH
        }
    
    def validate_calibration(self, tolerance: float = 2.0) -> Tuple[bool, dict]:
        """
        Validate calibration by projecting reference points.
        
        Args:
            tolerance: Acceptable error in meters
        
        Returns:
            (is_valid, stats): Validity flag and error statistics
        """
        
        image_pts = np.array(self.reference_keypoints_image, dtype=np.float32)
        world_pts_expected = np.array(self.reference_keypoints_world, dtype=np.float32)
        
        # Reproject reference points
        world_pts_reprojected = self.batch_image_to_world(image_pts, frame_homography=None)
        
        # Compute errors
        errors = np.linalg.norm(
            world_pts_reprojected - world_pts_expected,
            axis=1
        )
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        is_valid = max_error < tolerance
        
        stats = {
            'mean_error_m': mean_error,
            'max_error_m': max_error,
            'num_points': len(image_pts),
            'valid': is_valid
        }
        
        if not is_valid:
            logger.warning(
                f"Calibration validation failed: max_error={max_error:.2f}m "
                f"(tolerance={tolerance}m)"
            )
        
        return is_valid, stats


def _affine_inverse(affine_matrix: np.ndarray) -> np.ndarray:
    """
    Compute inverse of 2×3 affine transformation.
    
    Affine: [x', y'] = [A|b] @ [x, y, 1]^T
    where A is 2×2 and b is 2×1
    
    Inverse: [x, y] = A^-1 @ ([x', y'] - b)
    
    Returns as 2×3 matrix for consistency.
    
    Args:
        affine_matrix: 2×3 affine matrix
    
    Returns:
        2×3 inverse affine matrix
    """
    
    A = affine_matrix[:, :2]
    b = affine_matrix[:, 2]
    
    A_inv = np.linalg.inv(A)
    b_inv = -A_inv @ b
    
    return np.hstack([A_inv, b_inv.reshape(2, 1)])


def create_pitch_overlay(
    frame_shape: Tuple[int, int],
    projector: WorldProjector,
    frame_homography: Optional[np.ndarray] = None,
    grid_spacing: float = 5.0
) -> np.ndarray:
    """
    Create an overlay showing the pitch grid on the frame.
    
    Useful for visualization and validation.
    
    Args:
        frame_shape: (height, width)
        projector: WorldProjector instance
        frame_homography: Optional homography for current frame
        grid_spacing: Grid spacing in meters
    
    Returns:
        Overlay image (same shape as frame)
    """
    
    overlay = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
    
    # Generate world grid
    grid_world = PitchModel.get_grid_keypoints(
        spacing_x=grid_spacing,
        spacing_y=grid_spacing
    )
    
    # Project grid to image space
    for name, world_pt in grid_world.items():
        try:
            img_pt = projector.world_to_image(world_pt, frame_homography)
            
            # Clip to frame bounds
            x, y = int(img_pt[0]), int(img_pt[1])
            if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
        except Exception:
            pass  # Skip if projection fails
    
    return overlay
