# src/spatial/calibration_tool.py

"""
Interactive reference frame calibration utility.

Allows manual annotation of reference frame keypoints to establish
image → world coordinate mapping.

Usage:
    python -c "from spatial.calibration_tool import CalibratorUI; \
    ui = CalibratorUI(); ui.run('path/to/reference_frame.png')"
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ReferenceFrameCalibrator:
    """
    Calibrates reference frame by annotating keypoint correspondences.
    
    User clicks points in image and enters corresponding world coordinates (meters).
    
    Saves calibration for use in WorldProjector.
    """
    
    def __init__(self):
        """Initialize calibrator."""
        
        self.image: Optional[np.ndarray] = None
        self.image_points: List[Tuple[float, float]] = []
        self.world_points: List[Tuple[float, float]] = []
        
        self.current_point: Optional[Tuple[int, int]] = None
        self.active = False
    
    def load_frame(self, frame_path: str) -> None:
        """
        Load reference frame.
        
        Args:
            frame_path: Path to image file
        """
        
        self.image = cv2.imread(frame_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {frame_path}")
        
        logger.info(f"Loaded frame: {self.image.shape}")
    
    def add_keypoint(
        self,
        image_point: Tuple[float, float],
        world_point: Tuple[float, float]
    ) -> None:
        """
        Add correspondence between image and world coordinates.
        
        Args:
            image_point: (x, y) in pixels
            world_point: (X, Y) in meters
        """
        
        self.image_points.append(image_point)
        self.world_points.append(world_point)
        
        logger.info(f"Added keypoint: {image_point} → {world_point}")
    
    def get_affine_transform(self) -> Tuple[np.ndarray, dict]:
        """
        Compute affine transformation from image to world.
        
        Returns:
            (affine_matrix, stats)
        """
        
        if len(self.image_points) < 3:
            raise ValueError("Need at least 3 keypoints for affine transform")
        
        src = np.array(self.image_points[:3], dtype=np.float32)
        dst = np.array(self.world_points[:3], dtype=np.float32)
        
        affine = cv2.getAffineTransform(src, dst)
        
        # Validate on remaining points
        if len(self.image_points) > 3:
            test_src = np.array(self.image_points[3:], dtype=np.float32)
            test_dst = np.array(self.world_points[3:], dtype=np.float32)
            
            test_src_homog = np.hstack([test_src, np.ones((len(test_src), 1))])
            predicted_dst = (affine @ test_src_homog.T).T
            
            errors = np.linalg.norm(predicted_dst - test_dst, axis=1)
            
            stats = {
                'mean_error_m': float(np.mean(errors)),
                'max_error_m': float(np.max(errors)),
                'rmse': float(np.sqrt(np.mean(errors**2)))
            }
        else:
            stats = {}
        
        return affine, stats
    
    def get_homography_transform(self) -> Tuple[np.ndarray, dict]:
        """
        Compute homography transformation from image to world.
        
        Returns:
            (homography_matrix, stats)
        """
        
        if len(self.image_points) < 4:
            raise ValueError("Need at least 4 keypoints for homography")
        
        src = np.array(self.image_points[:4], dtype=np.float32)
        dst = np.array(self.world_points[:4], dtype=np.float32)
        
        H, _ = cv2.findHomography(src, dst)
        
        # Validate on remaining points
        if len(self.image_points) > 4:
            test_src = np.array(self.image_points[4:], dtype=np.float32)
            test_dst = np.array(self.world_points[4:], dtype=np.float32)
            
            predicted_dst = cv2.perspectiveTransform(
                test_src.reshape(-1, 1, 2),
                H
            ).reshape(-1, 2)
            
            errors = np.linalg.norm(predicted_dst - test_dst, axis=1)
            
            stats = {
                'mean_error_m': float(np.mean(errors)),
                'max_error_m': float(np.max(errors)),
                'rmse': float(np.sqrt(np.mean(errors**2)))
            }
        else:
            stats = {}
        
        return H, stats
    
    def save_calibration(self, output_path: str) -> None:
        """
        Save calibration to JSON file.
        
        Args:
            output_path: Path to output JSON
        """
        
        calibration = {
            'image_points': self.image_points,
            'world_points': self.world_points,
            'num_keypoints': len(self.image_points)
        }
        
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        logger.info(f"Saved calibration to {output_path}")
    
    @staticmethod
    def load_calibration(input_path: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Load calibration from JSON file.
        
        Args:
            input_path: Path to calibration JSON
        
        Returns:
            (image_points, world_points)
        """
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        image_points = [tuple(p) for p in data['image_points']]
        world_points = [tuple(p) for p in data['world_points']]
        
        return image_points, world_points
    
    def __repr__(self) -> str:
        return f"<Calibrator keypoints={len(self.image_points)}>"


class InteractiveCalibratorUI:
    """
    Simple CLI-based calibrator (non-GUI).
    
    Prompts user to enter pixel coordinates and world coordinates interactively.
    """
    
    def __init__(self):
        """Initialize CLI calibrator."""
        
        self.calibrator = ReferenceFrameCalibrator()
    
    def run_cli(self, frame_path: str, output_path: str) -> None:
        """
        Run interactive CLI calibration.
        
        Args:
            frame_path: Path to reference frame image
            output_path: Path to save calibration JSON
        """
        
        self.calibrator.load_frame(frame_path)
        
        print("\n=== Reference Frame Calibration ===")
        print(f"Loaded frame: {frame_path}")
        print("\nEnter keypoint correspondences.")
        print("Format: x y X Y (pixel coords, then world coords in meters)")
        print("Enter 'done' when finished.\n")
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if user_input.lower() == 'done':
                    break
                
                parts = user_input.split()
                if len(parts) != 4:
                    print("Invalid format. Use: x y X Y")
                    continue
                
                x, y, X, Y = map(float, parts)
                self.calibrator.add_keypoint((x, y), (X, Y))
                
            except ValueError:
                print("Invalid input. Use: x y X Y (floats)")
                continue
        
        # Save calibration
        self.calibrator.save_calibration(output_path)
        
        # Print stats
        print(f"\nCalibration saved: {len(self.calibrator.image_points)} keypoints")
        
        # Compute and display affine transform
        try:
            affine, stats = self.calibrator.get_affine_transform()
            print(f"Affine transform computed:")
            print(f"  Mean error: {stats.get('mean_error_m', 0):.2f}m")
            print(f"  Max error: {stats.get('max_error_m', 0):.2f}m")
        except Exception as e:
            print(f"Affine transform failed: {e}")


class ClickableCalibrator:
    """
    Mouse-clickable calibrator for easier interaction.
    
    Shows image, user clicks to select points, enters world coordinates.
    """
    
    def __init__(self):
        """Initialize clickable calibrator."""
        
        self.calibrator = ReferenceFrameCalibrator()
        self.pending_world_input = False
        self.last_click: Optional[Tuple[int, int]] = None
    
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: int) -> None:
        """
        Mouse click callback.
        
        Args:
            event: cv2 mouse event
            x, y: Mouse coordinates
            flags: Modifier flags
            param: User parameter
        """
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_click = (x, y)
            self.pending_world_input = True
            print(f"Clicked: ({x}, {y}). Enter world coordinates: X Y")
    
    def run_interactive(self, frame_path: str, output_path: str) -> None:
        """
        Run interactive GUI calibration.
        
        Args:
            frame_path: Path to reference frame
            output_path: Path to save calibration
        """
        
        self.calibrator.load_frame(frame_path)
        
        window_name = 'Reference Frame Calibrator'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n=== Interactive Reference Frame Calibration ===")
        print("Click on the image to select points.")
        print("Enter 'q' to quit and save.")
        
        while True:
            # Display image with marked points
            display = self.calibrator.image.copy()
            
            # Draw marked points
            for i, (x, y) in enumerate(self.calibrator.image_points):
                cv2.circle(display, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(display, str(i), (int(x) + 5, int(y) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow(window_name, display)
            
            # Handle input
            if self.pending_world_input:
                try:
                    world_input = input("> ").strip()
                    
                    if world_input.lower() == 'q':
                        break
                    
                    X, Y = map(float, world_input.split())
                    self.calibrator.add_keypoint(self.last_click, (X, Y))
                    self.pending_world_input = False
                
                except ValueError:
                    print("Invalid format. Use: X Y (floats)")
                    continue
            else:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
        
        cv2.destroyAllWindows()
        
        # Save
        self.calibrator.save_calibration(output_path)
        print(f"\nCalibration saved: {len(self.calibrator.image_points)} keypoints")


# Convenience functions
def create_sample_calibration(output_path: str) -> None:
    """
    Create a sample calibration file for testing.
    
    Assumes standard 105m × 68m pitch.
    
    Args:
        output_path: Path to save calibration
    """
    
    # Sample keypoints: corners of pitch
    image_points = [
        (100, 500),      # Bottom-left
        (1800, 500),     # Bottom-right
        (100, 100),      # Top-left
        (1800, 100),     # Top-right
    ]
    
    world_points = [
        (0.0, 0.0),           # Bottom-left (0, 0)
        (105.0, 0.0),         # Bottom-right (105, 0)
        (0.0, 68.0),          # Top-left (0, 68)
        (105.0, 68.0),        # Top-right (105, 68)
    ]
    
    calibrator = ReferenceFrameCalibrator()
    for img_pt, world_pt in zip(image_points, world_points):
        calibrator.add_keypoint(img_pt, world_pt)
    
    calibrator.save_calibration(output_path)
    print(f"Sample calibration created: {output_path}")


if __name__ == '__main__':
    # Test: Create sample calibration
    create_sample_calibration('/tmp/sample_calibration.json')
