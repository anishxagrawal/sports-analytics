# src/spatial/homography_estimator.py

"""
Homography estimation for camera motion compensation.

Estimates 2D planar homography between consecutive frames using:
1. Feature matching (from pitch lines or generic features)
2. RANSAC fitting for robustness
3. Geometric validation
4. Confidence scoring
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HomographyResult:
    """Result of homography estimation."""
    
    H: Optional[np.ndarray]  # 3×3 homography matrix
    confidence: float  # [0, 1]
    is_valid: bool  # Passes validation checks
    inlier_ratio: float  # Fraction of matches that are inliers
    reprojection_error: float  # Mean error for inliers (pixels)
    num_matches: int  # Total number of matches
    num_inliers: int  # Number of RANSAC inliers
    
    def __repr__(self) -> str:
        return (
            f"<HomographyResult conf={self.confidence:.2f} "
            f"valid={self.is_valid} inliers={self.inlier_ratio:.1%}>"
        )


def estimate_homography_ransac(
    src_points: np.ndarray,  # N×2, reference points
    dst_points: np.ndarray,  # N×2, current frame points
    max_reprojection_error: float = 5.0,
    confidence_level: float = 0.99,
    min_inliers: int = 4
) -> HomographyResult:
    """
    Estimate homography using RANSAC.
    
    Finds H such that: dst_points ≈ apply_homography(H, src_points)
    
    Args:
        src_points: Reference frame keypoints (N×2)
        dst_points: Current frame keypoints (N×2)
        max_reprojection_error: Threshold for inlier classification (pixels)
        confidence_level: RANSAC confidence (e.g., 0.99)
        min_inliers: Minimum required inliers
    
    Returns:
        HomographyResult with H matrix and validation metrics
    """
    
    if len(src_points) < 4 or len(dst_points) < 4:
        return HomographyResult(
            H=None, confidence=0.0, is_valid=False,
            inlier_ratio=0.0, reprojection_error=np.inf,
            num_matches=len(src_points), num_inliers=0
        )
    
    if len(src_points) != len(dst_points):
        return HomographyResult(
            H=None, confidence=0.0, is_valid=False,
            inlier_ratio=0.0, reprojection_error=np.inf,
            num_matches=len(src_points), num_inliers=0
        )
    
    # Convert to float32
    src_points = np.asarray(src_points, dtype=np.float32)
    dst_points = np.asarray(dst_points, dtype=np.float32)
    
    try:
        # OpenCV's findHomography with RANSAC
        H, inlier_mask = cv2.findHomography(
            dst_points,
            src_points,
            cv2.RANSAC,
            ransacReprojThreshold=max_reprojection_error
        )
    except cv2.error as e:
        logger.warning(f"cv2.findHomography failed: {e}")
        return HomographyResult(
            H=None, confidence=0.0, is_valid=False,
            inlier_ratio=0.0, reprojection_error=np.inf,
            num_matches=len(src_points), num_inliers=0
        )
    
    if H is None or inlier_mask is None:
        return HomographyResult(
            H=None, confidence=0.0, is_valid=False,
            inlier_ratio=0.0, reprojection_error=np.inf,
            num_matches=len(src_points), num_inliers=0
        )
    
    # Extract inliers
    inlier_mask = inlier_mask.ravel().astype(bool)
    num_inliers = np.sum(inlier_mask)
    inlier_ratio = num_inliers / len(src_points)
    
    # Compute reprojection error for inliers
    if num_inliers >= min_inliers:
        src_inliers = src_points[inlier_mask]
        dst_inliers = dst_points[inlier_mask]
        
        # Reproject: dst → src via H
        src_reprojected = cv2.perspectiveTransform(
            dst_inliers.reshape(-1, 1, 2),
            H
        ).reshape(-1, 2)
        
        reprojection_errors = np.linalg.norm(
            src_reprojected - src_inliers,
            axis=1
        )
        mean_reprojection_error = np.mean(reprojection_errors)
    else:
        mean_reprojection_error = np.inf
    
    # Validate homography
    confidence, is_valid = _validate_homography(
        H, inlier_ratio, mean_reprojection_error, num_inliers, min_inliers
    )
    
    return HomographyResult(
        H=H if is_valid else None,
        confidence=confidence,
        is_valid=is_valid,
        inlier_ratio=inlier_ratio,
        reprojection_error=mean_reprojection_error,
        num_matches=len(src_points),
        num_inliers=num_inliers
    )


def _validate_homography(
    H: np.ndarray,
    inlier_ratio: float,
    mean_reprojection_error: float,
    num_inliers: int,
    min_inliers: int
) -> Tuple[float, bool]:
    """
    Validate homography for geometric plausibility.
    
    Checks:
    1. Sufficient inlier ratio (>60%)
    2. Reasonable reprojection error (<10 pixels)
    3. Homography doesn't violate camera model
    4. Sufficient number of inliers
    
    Args:
        H: Homography matrix (3×3)
        inlier_ratio: Fraction of inliers
        mean_reprojection_error: Mean reprojection error (pixels)
        num_inliers: Absolute number of inliers
        min_inliers: Minimum required inliers
    
    Returns:
        (confidence, is_valid): Confidence in [0,1] and boolean validity
    """
    
    # Check 1: Sufficient inlier ratio
    MIN_INLIER_RATIO = 0.6
    if inlier_ratio < MIN_INLIER_RATIO:
        return 0.0, False
    
    # Check 2: Reasonable reprojection error
    MAX_REPROJECTION_ERROR = 10.0  # pixels
    if mean_reprojection_error > MAX_REPROJECTION_ERROR:
        return 0.0, False
    
    # Check 3: Sufficient absolute inliers
    if num_inliers < min_inliers:
        return 0.0, False
    
    # Check 4: Homography doesn't flip orientation
    det_H = np.linalg.det(H)
    if det_H < 0:
        return 0.0, False  # Orientation flipped
    
    # Check 5: Reasonable scale (not extreme zoom)
    # Extract scale from H via SVD
    U, S, Vt = np.linalg.svd(H[:2, :2])
    scales = S
    scale_ratio = np.max(scales) / (np.min(scales) + 1e-8)
    
    MAX_SCALE_RATIO = 5.0  # Extreme zoom
    if scale_ratio > MAX_SCALE_RATIO:
        return 0.0, False
    
    # If all checks pass, compute confidence
    # Confidence = 1 - normalized_error
    normalized_error = mean_reprojection_error / MAX_REPROJECTION_ERROR
    confidence = max(0.0, 1.0 - normalized_error)
    confidence *= inlier_ratio  # Weight by inlier ratio
    
    return confidence, True


def apply_homography(
    points: np.ndarray,  # N×2
    H: np.ndarray  # 3×3
) -> np.ndarray:
    """
    Apply homography transformation to 2D points.
    
    Args:
        points: N×2 array of (x, y) coordinates
        H: 3×3 homography matrix
    
    Returns:
        N×2 array of transformed points
    """
    
    points = np.asarray(points, dtype=np.float32)
    
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    # Vectorized: points ≈ apply_homography(H, points)
    points_homog = np.hstack([points, np.ones((len(points), 1))])
    transformed_homog = (H @ points_homog.T).T
    transformed = transformed_homog[:, :2] / transformed_homog[:, 2:3]
    
    return transformed


def compose_homographies(
    H1: np.ndarray,
    H2: np.ndarray
) -> np.ndarray:
    """
    Compose two homographies: H_total = H1 @ H2.
    
    If you have:
    - H1: transforms frame1 → frame0
    - H2: transforms frame2 → frame1
    
    Then H_total transforms frame2 → frame0.
    
    Args:
        H1, H2: 3×3 homography matrices
    
    Returns:
        Composed homography H1 @ H2
    """
    
    return H1 @ H2


def inverse_homography(H: np.ndarray) -> np.ndarray:
    """
    Compute inverse of homography.
    
    Args:
        H: 3×3 homography matrix
    
    Returns:
        Inverse homography
    """
    
    return np.linalg.inv(H)


def homography_to_affine(H: np.ndarray) -> np.ndarray:
    """
    Extract 2×3 affine approximation from homography.
    
    Useful for visualization or when affine is sufficient.
    
    Args:
        H: 3×3 homography
    
    Returns:
        2×3 affine matrix
    """
    
    return H[:2, :]


def detect_homography_discontinuity(
    H_prev: np.ndarray,
    H_curr: np.ndarray,
    rotation_threshold: float = 0.5  # radians
) -> bool:
    """
    Detect camera cut or sudden motion (discontinuity).
    
    Args:
        H_prev, H_curr: Consecutive homographies
        rotation_threshold: Threshold for rotation angle (radians)
    
    Returns:
        True if discontinuity detected (likely camera cut)
    """
    
    # Compute relative homography
    H_delta = H_curr @ np.linalg.inv(H_prev)
    
    # Extract rotation component via SVD
    U, S, Vt = np.linalg.svd(H_delta[:2, :2])
    R_approx = U @ Vt
    
    # Compute rotation angle
    trace = np.trace(R_approx)
    rotation_angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    
    return rotation_angle > rotation_threshold


def decompose_homography(H: np.ndarray) -> dict:
    """
    Decompose homography into components for analysis.
    
    Args:
        H: 3×3 homography matrix
    
    Returns:
        Dictionary with decomposition details
    """
    
    # Extract rotation and scale from upper-left 2×2
    M = H[:2, :2]
    
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt  # Pure rotation
    scale_factors = S  # Singular values = scale
    
    # Translation
    t = H[:2, 2]
    
    # Perspective component
    p = H[2, :2]
    
    return {
        'rotation_matrix': R,
        'scale_factors': scale_factors,
        'translation': t,
        'perspective': p,
        'det': np.linalg.det(H),
        'condition_number': np.linalg.cond(H)
    }
