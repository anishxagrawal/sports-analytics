# src/main.py

"""
Main orchestration script for sports analytics pipeline.

Connects all pipeline components and runs frame-by-frame processing
on video input with commentary generation.
"""

import cv2
import numpy as np
from pathlib import Path

from config.models import get_model
from core.video import VideoReader
from core.detector import YOLODetector
from core.tracker import Tracker
from entities.entity_manager import EntityManager
from analytics.events import detect_player_events
from commentary.engine import CommentaryEngine
from commentary.prompt_builder import PromptBuilder
from commentary.llm_adapter import LLMAdapter
from spatial.world_projection_pipeline import WorldProjectionPipeline
from spatial.calibration_tool import ReferenceFrameCalibrator
from visualization.pitch_overlay import PitchOverlay
from visualization.object_renderer import draw_players_with_metrics, draw_ball_with_metrics


def main():
    """Run the sports analytics pipeline on a video file."""
    
    # Get project root directory (parent of src/)
    project_root = Path(__file__).parent.parent
    
    # Configuration
    video_path = project_root / "data/inputs/test_video_6.mp4"
    model_path = str(get_model("ball"))
    conf_threshold = 0.12
    allowed_classes = {0, 1, 2}
    
    # Per-class confidence thresholds
    PLAYER_MIN_CONF = 0.20
    BALL_MIN_CONF = 0.12
    REFEREE_MIN_CONF = 0.20
    
    # Class name mapping for debug visualization
    class_names = {
        0: "Player",
        1: "Ball",
        2: "Referee"
    }
    
    print("Initializing pipeline...")
    
    print("Loading video...")
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
    
    if not writer.isOpened():
        print("ERROR: Failed to open video writer!")
        return

    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        allowed_classes=allowed_classes,
        device="cpu"
    )
    
    tracker = Tracker(frame_rate=fps, track_thresh=0.25)
    entity_manager = EntityManager()
    
    # Initialize world projection pipeline for real-world metrics (m/s)
    print("Initializing world projection pipeline...")
    video_reader_temp = VideoReader(str(video_path))
    reference_frame, _ = next(video_reader_temp)
    video_reader_temp.release()
    
    # Load calibration (use sample if not found)
    try:
        img_pts, world_pts = ReferenceFrameCalibrator.load_calibration(
            str(project_root / "calibration.json")
        )
        print(f"Loaded calibration with {len(img_pts)} keypoints")
    except FileNotFoundError:
        print("WARNING: calibration.json not found. Using sample calibration.")
        world_pts = [(0, 0), (105, 0), (0, 68), (105, 68)]
    
    world_pipeline = WorldProjectionPipeline(
        reference_frame=reference_frame,
        reference_keypoints_world=world_pts,
        fps=fps,
        enable_homography=True,
        enable_velocity=True
    )
    
    # --- Pitch overlay integration ---
    # Initialize debug visualization overlay
    pitch_overlay = PitchOverlay(
        width=300,
        height=200,
        position="top-right",
        margin=20,
        alpha=0.85,
        enabled=True
    )
    # --- End pitch overlay integration ---
    
    commentary_engine = CommentaryEngine(cooldown_seconds=5.0)
    prompt_builder = PromptBuilder()
    llm_adapter = LLMAdapter()
    
    print(f"Processing video at {fps} FPS...\n")
    
    # Frame-by-frame processing loop
    frame_count = 0
    for frame, metadata in video_reader:
        frame_count += 1
        frame_idx = metadata['frame_idx']
        timestamp = metadata['timestamp']
        
        if frame_count % 50 == 0:
            print(f"Frame {frame_count}...")
        
        detections = detector.detect(frame)
        
        # Per-class confidence filtering
        filtered_detections = []
        for det in detections:
            class_id = det["class_id"]
            confidence = det["confidence"]
            
            if class_id == 0 and confidence >= PLAYER_MIN_CONF:
                filtered_detections.append(det)
            elif class_id == 1 and confidence >= BALL_MIN_CONF:
                filtered_detections.append(det)
            elif class_id == 2 and confidence >= REFEREE_MIN_CONF:
                filtered_detections.append(det)
        
        detections = filtered_detections

        result = tracker.update(detections, frame, frame_idx)
        tracks = result["tracks"]
        ball_state = result["ball"]

        # Process through world projection pipeline
        enriched_tracks = world_pipeline.process_frame(frame, tracks, frame_idx)

        if ball_state.get("position") is not None:
            ball_pos = ball_state["position"]
            ball_bbox = (
                ball_pos[0] - 10,
                ball_pos[1] - 10,
                ball_pos[0] + 10,
                ball_pos[1] + 10
            )
            
            ball_detection = {
                'bbox': ball_bbox,
                'class_id': 1,
                'confidence': ball_state.get("confidence", 0.9),
                'track_id': -1
            }
            
            enriched_ball = world_pipeline.process_frame(frame, [ball_detection], frame_idx)
            
            if enriched_ball:
                ball_state['world_position'] = enriched_ball[0].get('world_position')
                ball_state['velocity'] = enriched_ball[0].get('velocity')
                ball_state['speed'] = enriched_ball[0].get('speed')

        entity_manager.update(enriched_tracks, frame_idx, timestamp)
        entity_manager.update_ball(ball_state, frame_idx)
        
        # Draw player visualization with circles, arrows, and metrics
        draw_players_with_metrics(frame, entity_manager.get_active_players(), fps)
        
        # Draw ball visualization with circle, arrow, and metrics
        draw_ball_with_metrics(frame, entity_manager.ball, fps)
        
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

        # --- Pitch overlay integration ---
        # Collect field positions for visualization
        players_for_viz = []
        for player in entity_manager.get_active_players():
            # Get latest field position from deque
            field_pos = None
            field_pos_anchored = None
            
            if hasattr(player, 'field_positions_anchored') and player.field_positions_anchored:
                field_pos_anchored = player.field_positions_anchored[-1]
            
            if hasattr(player, 'field_positions') and player.field_positions:
                field_pos = player.field_positions[-1]
            
            players_for_viz.append({
                'field_position': field_pos,
                'field_position_anchored': field_pos_anchored
            })
        
        ball_for_viz = None
        if entity_manager.ball.is_visible():
            field_pos = None
            field_pos_anchored = None
            
            if hasattr(entity_manager.ball, 'field_position_anchored') and entity_manager.ball.field_position_anchored:
                field_pos_anchored = entity_manager.ball.field_position_anchored
            
            if hasattr(entity_manager.ball, 'field_position') and entity_manager.ball.field_position:
                field_pos = entity_manager.ball.field_position
            
            ball_for_viz = {
                'field_position': field_pos,
                'field_position_anchored': field_pos_anchored
            }
        
        # Render overlay onto frame
        frame = pitch_overlay.render(frame, players_for_viz, ball_for_viz)
        # --- End pitch overlay integration ---
        
        # Print world-space metrics (real-world speed in m/s)
        if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
            for track in enriched_tracks:
                if track.get('speed') is not None:
                    track_id = track.get('track_id', 'unknown')
                    world_pos = track.get('world_position', (0, 0))
                    speed = track.get('speed', 0)
                    if speed > 0:
                        print(f"  Player {track_id}: ({world_pos[0]:.1f}m, {world_pos[1]:.1f}m) @ {speed:.2f}m/s")

        # Write frame to output video
        writer.write(frame)

    # ✅ Proper cleanup (ONCE)
    video_reader.release()
    writer.release()
    print("\nProcessing complete.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()