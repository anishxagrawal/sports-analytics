# src/core/tracker.py

"""
Multi-object tracking module for sports analytics system.

Provides a clean ByteTrack wrapper that maintains object identities across
frames using detection inputs and returns tracker-agnostic track dictionaries.

Notes:
    - Tracker is stateful across frames
    - Frame rate must match the actual video FPS for optimal performance
    - ByteTrack tracking is CLASS-AGNOSTIC: it tracks all objects together 
      regardless of class_id. Track IDs are unique integers that persist 
      across frames for the same object.
    - Ball tracking uses CLASS-SPECIFIC logic: detections with class_id == 1
      (Ball class) are processed separately by BallTracker for 
      stabilization and prediction.
    - ByteTrack output format: [x1, y1, x2, y2, track_id, conf, class_id, index]
"""

from typing import List, Dict, Any
import numpy as np
from boxmot import ByteTrack
from core.ball_tracker import BallTracker


class Tracker:
    """
    Multi-object tracker using ByteTrack with specialized ball tracking.
    
    Maintains object identities across frames by associating detections using
    motion and appearance cues. Stateful across frame updates.
    
    ByteTrack tracking is class-agnostic: all objects are tracked together
    in a single tracking space. However, ball detections (class_id == 1) are
    additionally processed by BallTracker for stabilization and prediction.
    
    Usage:
        tracker = Tracker(frame_rate=30.0)
        for frame_idx, frame_detections in enumerate(detection_stream):
            result = tracker.update(frame_detections, frame, frame_idx)
            for track in result['tracks']:
                track_id = track['track_id']
                x1, y1, x2, y2 = track['bbox']
            ball_state = result['ball']
    
    Attributes:
        frame_rate: Video frame rate (FPS) for motion prediction
        track_thresh: Detection confidence threshold for track initialization
        track_buffer: Number of frames to keep lost tracks alive
        match_thresh: IOU threshold for matching detections to tracks
    """
    
    # Ball class ID for this model
    BALL_CLASS_ID = 1
    
    # ByteTrack output array indices (documented for maintainability)
    _BYTE_X1 = 0
    _BYTE_Y1 = 1
    _BYTE_X2 = 2
    _BYTE_Y2 = 3
    _BYTE_TRACK_ID = 4
    _BYTE_CONF = 5
    _BYTE_CLASS_ID = 6
    _BYTE_INDEX = 7
    _BYTE_EXPECTED_COLS = 8
    
    def __init__(
        self,
        frame_rate: float,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8
    ):
        """
        Initialize ByteTrack-based tracker.
        
        Args:
            frame_rate: Video frame rate in FPS (must match actual video FPS)
            track_thresh: Minimum detection confidence to start a new track
            track_buffer: Number of frames to buffer lost tracks before removal
            match_thresh: IOU threshold for associating detections with tracks
        
        Raises:
            ValueError: If frame_rate is invalid
        """
        if frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive, got {frame_rate}")
        
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self._tracker = ByteTrack(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate
        )
        
        self.ball_tracker = BallTracker()
    
    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: np.ndarray,
        frame_index: int
    ) -> Dict[str, Any]:
        """
        Update tracker with new frame detections.
        
        Args:
            detections: List of detection dictionaries, each containing:
                - bbox: tuple (x1, y1, x2, y2) in pixel coordinates
                - confidence: float confidence score
                - class_id: int class identifier
            frame: Frame array for ByteTrack
            frame_index: Current frame index from video
        
        Returns:
            Dictionary containing:
                - tracks: List of tracked object dictionaries, each containing:
                    - track_id: int unique track identifier
                    - bbox: tuple (x1, y1, x2, y2) in pixel coordinates
                    - confidence: float detection confidence
                    - class_id: int class identifier
                - ball: Ball tracking state dictionary from BallTracker
            
            Returns empty tracks list if no active tracks.
        
        Notes:
            - Tracker maintains state across calls
            - Track IDs persist across frames for the same object
            - Lost tracks are kept alive for track_buffer frames
            - ByteTrack tracks all classes together; track_ids span all classes
            - Ball (class_id == 1) is additionally tracked by BallTracker
        """
        # Extract ball detection (class_id == 1)
        ball_detections = [det for det in detections if det['class_id'] == self.BALL_CLASS_ID]
        
        ball_position = None
        ball_confidence = None
        
        if ball_detections:
            # Use highest confidence ball detection
            best_ball = max(ball_detections, key=lambda d: d['confidence'])
            x1, y1, x2, y2 = best_ball['bbox']
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            ball_position = (center_x, center_y)
            ball_confidence = best_ball['confidence']
        
        # Update ball tracker
        ball_state = self.ball_tracker.update(
            position=ball_position,
            frame_index=frame_index,
            confidence=ball_confidence
        )
        
        # Update ByteTrack
        if len(detections) == 0:
            empty_dets = np.empty((0, 6), dtype=np.float32)
            self._tracker.update(empty_dets, frame)
            return {
                'tracks': [],
                'ball': ball_state
            }

        dets_array = np.array([
            [
                det['bbox'][0],
                det['bbox'][1],
                det['bbox'][2],
                det['bbox'][3],
                det['confidence'],
                det['class_id']
            ]
            for det in detections
        ], dtype=np.float32)
        
        tracks_output = self._tracker.update(dets_array, frame)
        
        tracked_objects = []
        
        if tracks_output is None or len(tracks_output) == 0:
            return {
                'tracks': tracked_objects,
                'ball': ball_state
            }
        
        for track in tracks_output:
            if len(track) < self._BYTE_EXPECTED_COLS:
                continue
            
            x1 = track[self._BYTE_X1]
            y1 = track[self._BYTE_Y1]
            x2 = track[self._BYTE_X2]
            y2 = track[self._BYTE_Y2]
            track_id = track[self._BYTE_TRACK_ID]
            conf = track[self._BYTE_CONF]
            class_id = track[self._BYTE_CLASS_ID]
            
            tracked_obj = {
                'track_id': int(track_id),
                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                'confidence': float(conf),
                'class_id': int(class_id)
            }
            
            tracked_objects.append(tracked_obj)
        
        return {
            'tracks': tracked_objects,
            'ball': ball_state
        }