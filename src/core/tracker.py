# src/core/tracker.py

"""
Multi-object tracking module for sports analytics system.

Provides a clean ByteTrack wrapper that maintains object identities across
frames using detection inputs and returns tracker-agnostic track dictionaries.

Notes:
    - Tracker is stateful across frames
    - Frame rate must match the actual video FPS for optimal performance
    - This tracker is CLASS-AGNOSTIC: it tracks all objects together regardless
      of class_id. If you need per-class tracking (e.g., separate track IDs for
      players vs. balls), instantiate one Tracker per class and filter detections
      before calling update().
    - Track IDs are unique integers that persist across frames for the same object
    - ByteTrack output format: [x1, y1, x2, y2, track_id, conf, class_id, index]
"""

from typing import List, Dict, Any
import numpy as np
from boxmot import ByteTrack


class Tracker:
    """
    Multi-object tracker using ByteTrack.
    
    Maintains object identities across frames by associating detections using
    motion and appearance cues. Stateful across frame updates.
    
    IMPORTANT: This tracker is class-agnostic. All objects are tracked together
    in a single tracking space. For per-class tracking, create separate Tracker
    instances and filter detections by class before updating.
    
    Usage:
        tracker = Tracker(frame_rate=30.0)
        for frame_detections in detection_stream:
            tracks = tracker.update(frame_detections)
            for track in tracks:
                track_id = track['track_id']
                x1, y1, x2, y2 = track['bbox']
    
    Attributes:
        frame_rate: Video frame rate (FPS) for motion prediction
        track_thresh: Detection confidence threshold for track initialization
        track_buffer: Number of frames to keep lost tracks alive
        match_thresh: IOU threshold for matching detections to tracks
    """
    
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
    
    def update(
    self,
    detections: List[Dict[str, Any]],
    frame: np.ndarray
) -> List[Dict[str, Any]]:

        """
        Update tracker with new frame detections.
        
        Args:
            detections: List of detection dictionaries, each containing:
                - bbox: tuple (x1, y1, x2, y2) in pixel coordinates
                - confidence: float confidence score
                - class_id: int class identifier
        
        Returns:
            List of tracked object dictionaries, each containing:
                - track_id: int unique track identifier
                - bbox: tuple (x1, y1, x2, y2) in pixel coordinates
                - confidence: float detection confidence
                - class_id: int class identifier
            
            Returns empty list if no active tracks.
        
        Notes:
            - Tracker maintains state across calls
            - Track IDs persist across frames for the same object
            - Lost tracks are kept alive for track_buffer frames
            - All classes are tracked together; track_ids span all classes
        """
        if len(detections) == 0:
            empty_dets = np.empty((0, 6), dtype=np.float32)
            self._tracker.update(empty_dets, frame)
            return []

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
            return tracked_objects
        
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
        
        return tracked_objects