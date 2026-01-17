"""
Main orchestration script for sports analytics pipeline.

Connects all pipeline components and runs frame-by-frame processing
on video input with commentary generation.
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

from core.video import VideoReader
from core.detector import YOLODetector
from core.tracker import Tracker
from entities.entity_manager import EntityManager
from analytics.events import detect_player_events
from analytics.teams.assigner import extract_jersey_color
from commentary.engine import CommentaryEngine
from commentary.prompt_builder import PromptBuilder
from commentary.llm_adapter import LLMAdapter


class TeamColorAnchors:
    """
    Persistent team color references to prevent label flipping.
    
    Uses evidence-based locking: anchors lock only when separation
    and stability thresholds are met.
    
    Design: Anchors are fixed after locking (no drift adaptation).
    This prioritizes stability over dynamic color adjustment.
    """
    def __init__(self, min_separation=30.0, stability_window=5):
        self.team_a_anchor = None
        self.team_b_anchor = None
        self.min_separation = min_separation  # Minimum LAB distance between teams
        self.stability_window = stability_window  # Frames of stable estimates required
        self.accumulated_centroids = []  # List of (centroid_0, centroid_1) pairs
        self.is_locked = False
        self._lock_logged = False  # Track if lock event has been logged
    
    def _sort_centroids(self, centroid_0, centroid_1):
        """
        Sort centroids deterministically to ensure consistent pairing.
        Uses L channel (lightness) as the sorting key.
        """
        if centroid_0[0] <= centroid_1[0]:  # L channel comparison
            return centroid_0, centroid_1
        else:
            return centroid_1, centroid_0
    
    def accumulate(self, centroid_0, centroid_1):
        """Accumulate centroids with deterministic ordering during warmup."""
        if not self.is_locked:
            # Enforce stable ordering to prevent pairing ambiguity
            sorted_0, sorted_1 = self._sort_centroids(centroid_0, centroid_1)
            self.accumulated_centroids.append((sorted_0, sorted_1))
            
            # Check if anchors can be locked based on evidence
            self._try_lock_anchors()
    
    def _try_lock_anchors(self):
        """Lock anchors only if separation and stability criteria are met."""
        if len(self.accumulated_centroids) < self.stability_window:
            return
        
        # Compute average anchors from recent frames
        recent_centroids = self.accumulated_centroids[-self.stability_window:]
        all_centroid_0 = np.array([c[0] for c in recent_centroids])
        all_centroid_1 = np.array([c[1] for c in recent_centroids])
        
        candidate_a = all_centroid_0.mean(axis=0)
        candidate_b = all_centroid_1.mean(axis=0)
        
        # Check separation: teams must be sufficiently different
        separation = np.linalg.norm(candidate_a - candidate_b)
        if separation < self.min_separation:
            return
        
        # Check stability: recent centroids should not vary too much
        variance_a = np.var(all_centroid_0, axis=0).mean()
        variance_b = np.var(all_centroid_1, axis=0).mean()
        
        # Require low variance (stable estimates)
        if variance_a > 100.0 or variance_b > 100.0:
            return
        
        # All criteria met: lock anchors
        self.team_a_anchor = candidate_a
        self.team_b_anchor = candidate_b
        self.is_locked = True
        
        # Log anchor lock event (once)
        if not self._lock_logged:
            print(f"\n[TEAMS LOCKED] Anchors established with separation={separation:.2f}")
            self._lock_logged = True
    
    def is_ready(self):
        """Check if anchors are locked and ready for use."""
        return self.is_locked
    
    def assign_to_nearest_team(self, color_feature):
        """
        Assign a color directly to the nearest team anchor.
        
        Anchors are fixed after locking - no drift adaptation.
        This design prioritizes stability over dynamic color adjustment.
        
        Returns team label ("A" or "B") or None if not ready.
        """
        if not self.is_locked:
            return None
        
        dist_to_a = np.linalg.norm(np.array(color_feature) - self.team_a_anchor)
        dist_to_b = np.linalg.norm(np.array(color_feature) - self.team_b_anchor)
        
        return "A" if dist_to_a < dist_to_b else "B"


def main():
    """Run the sports analytics pipeline on a video file."""
    
    # Get project root directory (parent of src/)
    project_root = Path(__file__).parent.parent
    
    # Configuration
    video_path = project_root / "data/inputs/test_video.mp4"
    model_path = "yolov8n.pt"
    conf_threshold = 0.5
    allowed_classes = {0}  # Person class
    
    # Relaxed confidence threshold for jersey color extraction
    JERSEY_CONFIDENCE_THRESHOLD = 0.25
    
    print("Initializing pipeline...")
    
    video_reader = VideoReader(str(video_path))
    fps = video_reader.fps

    # Output video writer
    output_path = project_root / "data/outputs/output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (video_reader.width, video_reader.height)
    )

    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        allowed_classes=allowed_classes,
        device="cpu"
    )
    
    tracker = Tracker(frame_rate=fps)
    entity_manager = EntityManager()
    
    commentary_engine = CommentaryEngine(cooldown_seconds=5.0)
    prompt_builder = PromptBuilder()
    llm_adapter = LLMAdapter()
    
    # Team color anchor management with evidence-based locking
    team_anchors = TeamColorAnchors(min_separation=30.0, stability_window=5)
    
    print(f"Processing video at {fps} FPS...\n")
    
    # Frame-by-frame processing loop
    for frame, metadata in video_reader:
        frame_idx = metadata['frame_idx']
        timestamp = metadata['timestamp']
        
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        entity_manager.update(tracks, frame_idx, timestamp)
        
        # Build track_id -> bbox mapping from current frame
        track_bbox_map = {track["track_id"]: track["bbox"] for track in tracks}
        
        # Team identification using jersey color
        active_players = entity_manager.get_active_players()
        
        if len(active_players) >= 2:
            # Collect jersey color features for all active players
            player_features = []
            player_list = []
            
            for player in active_players:
                # Match player to current frame bbox via track_id
                if player.track_id in track_bbox_map:
                    bbox = track_bbox_map[player.track_id]
                    result = extract_jersey_color(frame, bbox)
                    
                    # Relaxed confidence gating to allow more valid extractions
                    if result['valid'] and result['confidence'] > JERSEY_CONFIDENCE_THRESHOLD:
                        player_features.append(result['color'])
                        player_list.append(player)
            
            if len(player_features) >= 2:
                if not team_anchors.is_ready():
                    # Warmup phase: run KMeans to accumulate centroids
                    features_array = np.array(player_features)
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(features_array)
                    
                    # Compute cluster centroids
                    cluster_0_mask = cluster_labels == 0
                    cluster_1_mask = cluster_labels == 1
                    
                    cluster_0_centroid = features_array[cluster_0_mask].mean(axis=0)
                    cluster_1_centroid = features_array[cluster_1_mask].mean(axis=0)
                    
                    # Accumulate with stable ordering
                    team_anchors.accumulate(cluster_0_centroid, cluster_1_centroid)
                    
                    # Deterministic cluster-to-team mapping for this frame
                    sorted_0, sorted_1 = team_anchors._sort_centroids(cluster_0_centroid, cluster_1_centroid)
                    
                    # Decide which cluster index maps to Team A
                    if cluster_0_centroid[0] <= cluster_1_centroid[0]:
                        cluster_0_is_team_a = True
                    else:
                        cluster_0_is_team_a = False
                    
                    # Assign provisional team labels during warmup
                    for player, cluster_label in zip(player_list, cluster_labels):
                        if cluster_label == 0:
                            team_label = "A" if cluster_0_is_team_a else "B"
                        else:
                            team_label = "B" if cluster_0_is_team_a else "A"
                        player.record_team_vote(team_label)
                else:
                    # Anchors locked: assign each player directly to nearest anchor
                    # No KMeans needed - use direct distance-based assignment
                    for player, color_feature in zip(player_list, player_features):
                        team_label = team_anchors.assign_to_nearest_team(color_feature)
                        if team_label is not None:
                            player.record_team_vote(team_label)
        
        # Draw warmup status overlay for observability
        status_text = "TEAMS LOCKED" if team_anchors.is_ready() else "WARMUP"
        status_color = (0, 255, 0) if team_anchors.is_ready() else (0, 165, 255)  # Green or Orange
        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )
        
        # Draw tracking results with team information
        for track in tracks:
            x1, y1, x2, y2 = map(int, track["bbox"])
            track_id = track["track_id"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get player entity and team information
            player = entity_manager.get_player_by_track_id(track_id)
            if player and player.team_id:
                label_text = f"ID {track_id} | Team {player.team_id}"
            else:
                label_text = f"ID {track_id} | Team UNK"
            
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Event detection
        all_events = []
        for player in entity_manager.get_active_players():
            all_events.extend(detect_player_events(player, fps))
        
        # Commentary pipeline
        if all_events:
            intents = commentary_engine.process_events(all_events, timestamp)
            if intents:
                prompt = prompt_builder.build(intents)
                if prompt:
                    commentary = llm_adapter.generate(prompt)
                    if commentary:
                        print(f"[{timestamp:.2f}s] {commentary}")

        writer.write(frame)

    # ✅ Proper cleanup (ONCE)
    video_reader.release()
    writer.release()
    print("\nProcessing complete.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()