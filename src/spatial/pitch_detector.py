# src/spatial/pitch_detector.py

"""
Pitch line detection for camera calibration.

Detects white pitch lines in broadcast football footage using:
1. Color-based thresholding (white detection)
2. Morphological filtering
3. Hough line detection
4. Keypoint extraction (line intersections, endpoints)
"""

import cv2
import numpy as np
from typing import List, Tuple, NamedTuple, Optional
from dataclasses import dataclass


@dataclass
class Line:
    """Represents a detected pitch line."""
    
    endpoints: Tuple[Tuple[float, float], Tuple[float, float]]  # ((x1, y1), (x2, y2))
    length: float
    orientation: str  # 'horizontal', 'vertical', or 'diagonal'
    confidence: float  # [0, 1]
    
    def midpoint(self) -> Tuple[float, float]:
        """Return midpoint of line."""
        x1, y1 = self.endpoints[0]
        x2, y2 = self.endpoints[1]
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def as_array(self) -> np.ndarray:
        """Return as array [x1, y1, x2, y2]."""
        x1, y1 = self.endpoints[0]
        x2, y2 = self.endpoints[1]
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    def __repr__(self) -> str:
        return f"<Line len={self.length:.1f} orient={self.orientation} conf={self.confidence:.2f}>"


def detect_pitch_lines(
    frame: np.ndarray,
    blur_kernel: int = 5,
    morph_kernel_size: int = 5,
    hough_threshold: int = 50,
    min_line_length: int = 30,
    max_line_gap: int = 10,
    debug: bool = False
) -> List[Line]:
    """
    Detect white pitch lines in a football frame.
    
    Pipeline:
    1. Convert to HSV and mask white pixels
    2. Morphological filtering to reduce noise
    3. Hough line detection
    4. Post-process and classify orientation
    
    Args:
        frame: Input BGR image
        blur_kernel: Kernel size for Gaussian blur
        morph_kernel_size: Kernel size for morphological operations
        hough_threshold: Hough line detection threshold
        min_line_length: Minimum line length to keep
        max_line_gap: Maximum gap to connect line segments
        debug: If True, return debug mask
    
    Returns:
        List of detected Line objects
    """
    
    if frame is None or frame.size == 0:
        return []
    
    # Step 1: Color-based thresholding for white pixels
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # White in HSV: High V (brightness), Low S (saturation)
    # Range: V > 200, S < 30
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Step 2: Morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Optional: Gaussian blur to reduce noise
    if blur_kernel > 1 and blur_kernel % 2 == 1:
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Step 3: Hough line detection
    lines_raw = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines_raw is None:
        return []
    
    # Step 4: Post-process detected lines
    detected_lines = []
    
    for line in lines_raw:
        x1, y1, x2, y2 = line[0]
        
        # Compute length
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if length < min_line_length:
            continue
        
        # Classify orientation
        angle = np.arctan2(y2 - y1, x2 - x1)
        orientation = _classify_orientation(angle)
        
        # All detected lines get high confidence from Hough detector
        # (could refine by checking alignment with grid)
        confidence = 0.8
        
        detected_lines.append(
            Line(
                endpoints=((float(x1), float(y1)), (float(x2), float(y2))),
                length=length,
                orientation=orientation,
                confidence=confidence
            )
        )
    
    return detected_lines


def extract_keypoints_from_lines(
    lines: List[Line],
    frame_shape: Tuple[int, int],
    merge_distance: float = 15.0,
    include_endpoints: bool = True,
    include_intersections: bool = True
) -> List[Tuple[float, float]]:
    """
    Extract keypoints as line intersections and endpoints.
    
    Args:
        lines: List of detected Line objects
        frame_shape: (height, width) for bounds checking
        merge_distance: Distance threshold for merging nearby keypoints
        include_endpoints: Whether to include line endpoints
        include_intersections: Whether to include line-line intersections
    
    Returns:
        List of unique keypoints (x, y) in image space
    """
    
    keypoints = []
    
    # Option 1: Line intersections (grid structure)
    if include_intersections:
        h_lines = [l for l in lines if l.orientation == 'horizontal']
        v_lines = [l for l in lines if l.orientation == 'vertical']
        
        for h_line in h_lines:
            for v_line in v_lines:
                intersection = _line_intersection(h_line, v_line)
                if intersection is not None and _is_within_bounds(intersection, frame_shape):
                    keypoints.append(intersection)
    
    # Option 2: Line endpoints
    if include_endpoints:
        for line in lines:
            p1, p2 = line.endpoints
            if _is_within_bounds(p1, frame_shape):
                keypoints.append(p1)
            if _is_within_bounds(p2, frame_shape):
                keypoints.append(p2)
    
    # Option 3: Line midpoints
    for line in lines:
        mid = line.midpoint()
        if _is_within_bounds(mid, frame_shape):
            keypoints.append(mid)
    
    # Deduplicate by clustering nearby points
    if len(keypoints) > 0:
        keypoints = _cluster_keypoints(keypoints, merge_distance)
    
    return keypoints


def _classify_orientation(angle: float) -> str:
    """
    Classify line orientation based on angle.
    
    Args:
        angle: Angle in radians from arctan2
    
    Returns:
        'horizontal', 'vertical', or 'diagonal'
    """
    
    # Normalize angle to [0, π)
    angle_norm = angle % np.pi
    
    # Horizontal: near 0 or π
    if angle_norm < np.pi / 8 or angle_norm > 7 * np.pi / 8:
        return 'horizontal'
    
    # Vertical: near π/2
    if np.pi / 8 < angle_norm < 7 * np.pi / 8:
        if 3 * np.pi / 8 < angle_norm < 5 * np.pi / 8:
            return 'vertical'
    
    return 'diagonal'


def _line_intersection(
    line1: Line,
    line2: Line,
    tolerance: float = 1e-6
) -> Optional[Tuple[float, float]]:
    """
    Compute intersection of two lines (if they intersect).
    
    Uses parametric line representation:
    P = P1 + t * (P2 - P1)
    
    Args:
        line1, line2: Line objects
        tolerance: Tolerance for parallel lines
    
    Returns:
        Intersection point (x, y) or None if parallel
    """
    
    x1, y1 = line1.endpoints[0]
    x2, y2 = line1.endpoints[1]
    
    x3, y3 = line2.endpoints[0]
    x4, y4 = line2.endpoints[1]
    
    # Direction vectors
    dx1 = x2 - x1
    dy1 = y2 - y1
    
    dx2 = x4 - x3
    dy2 = y4 - y3
    
    # Check for parallel lines (cross product ~ 0)
    cross = dx1 * dy2 - dy1 * dx2
    if abs(cross) < tolerance:
        return None
    
    # Solve for intersection
    t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / cross
    
    x_int = x1 + t1 * dx1
    y_int = y1 + t1 * dy1
    
    return (x_int, y_int)


def _is_within_bounds(
    point: Tuple[float, float],
    frame_shape: Tuple[int, int],
    margin: int = 10
) -> bool:
    """
    Check if point is within frame bounds.
    
    Args:
        point: (x, y)
        frame_shape: (height, width)
        margin: Margin from edges
    
    Returns:
        True if within bounds
    """
    
    x, y = point
    height, width = frame_shape
    
    return margin <= x <= width - margin and margin <= y <= height - margin


def _cluster_keypoints(
    keypoints: List[Tuple[float, float]],
    distance_threshold: float
) -> List[Tuple[float, float]]:
    """
    Merge nearby keypoints using clustering.
    
    Args:
        keypoints: List of (x, y) coordinates
        distance_threshold: Distance for merging
    
    Returns:
        Deduplicated keypoints
    """
    
    if len(keypoints) == 0:
        return []
    
    keypoints_array = np.array(keypoints, dtype=np.float32)
    
    # Use a simple greedy clustering
    clustered = []
    used = set()
    
    for i, kp in enumerate(keypoints_array):
        if i in used:
            continue
        
        # Find all nearby keypoints
        distances = np.linalg.norm(keypoints_array - kp, axis=1)
        cluster = np.where(distances <= distance_threshold)[0]
        
        # Average them
        centroid = keypoints_array[cluster].mean(axis=0)
        clustered.append(tuple(centroid))
        
        # Mark as used
        used.update(cluster)
    
    return clustered


def visualize_pitch_lines(
    frame: np.ndarray,
    lines: List[Line],
    keypoints: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Visualize detected pitch lines and keypoints.
    
    Args:
        frame: Input BGR image
        lines: Detected lines
        keypoints: Optional keypoints to visualize
    
    Returns:
        Annotated frame
    """
    
    frame_vis = frame.copy()
    
    # Draw lines
    for line in lines:
        x1, y1 = line.endpoints[0]
        x2, y2 = line.endpoints[1]
        
        # Color by orientation
        if line.orientation == 'horizontal':
            color = (0, 255, 0)  # Green
        elif line.orientation == 'vertical':
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 0, 255)  # Red
        
        cv2.line(frame_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    # Draw keypoints
    if keypoints is not None:
        for x, y in keypoints:
            cv2.circle(frame_vis, (int(x), int(y)), 5, (255, 255, 0), -1)
            cv2.circle(frame_vis, (int(x), int(y)), 5, (0, 0, 0), 1)
    
    return frame_vis


def get_line_statistics(lines: List[Line]) -> dict:
    """
    Compute statistics about detected lines.
    
    Returns:
        Dictionary with counts and lengths
    """
    
    if len(lines) == 0:
        return {
            'total_lines': 0,
            'horizontal': 0,
            'vertical': 0,
            'diagonal': 0,
            'avg_length': 0.0
        }
    
    h_lines = [l for l in lines if l.orientation == 'horizontal']
    v_lines = [l for l in lines if l.orientation == 'vertical']
    d_lines = [l for l in lines if l.orientation == 'diagonal']
    
    lengths = [l.length for l in lines]
    
    return {
        'total_lines': len(lines),
        'horizontal': len(h_lines),
        'vertical': len(v_lines),
        'diagonal': len(d_lines),
        'avg_length': np.mean(lengths) if lengths else 0.0,
        'total_length': np.sum(lengths)
    }
