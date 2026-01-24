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


def main():
    """Run the sports analytics pipeline on a video file."""
    
    # Get project root directory (parent of src/)
    project_root = Path(__file__).parent.parent
    
    # Configuration
    video_path = project_root / "data/inputs/test_video_2.mp4"
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
    
    tracker = Tracker(frame_rate=fps, track_thresh=0.25)
    entity_manager = EntityManager()
    
    commentary_engine = CommentaryEngine(cooldown_seconds=5.0)
    prompt_builder = PromptBuilder()
    llm_adapter = LLMAdapter()
    
    print(f"Processing video at {fps} FPS...\n")
    
    # Frame-by-frame processing loop
    for frame, metadata in video_reader:
        frame_idx = metadata['frame_idx']
        timestamp = metadata['timestamp']
        
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

        # DEBUG: draw raw detections BEFORE tracking
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cls = det["class_id"]
            conf = det["confidence"]

            if cls == 1:
                color = (0, 0, 255)  # Ball: red
            elif cls == 0:
                color = (255, 0, 0)  # Player: blue
            elif cls == 2:
                color = (0, 255, 255)  # Referee: yellow
            else:
                color = (255, 255, 255)  # Unknown: white
            
            class_name = class_names.get(cls, str(cls))
            label = f"RAW {class_name} cls={cls} conf={conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )


        result = tracker.update(detections, frame, frame_idx)
        tracks = result["tracks"]
        ball_state = result["ball"]

        entity_manager.update(tracks, frame_idx, timestamp)
        entity_manager.update_ball(ball_state, frame_idx)
        
        # Draw tracking results
        for track in tracks:
            x1, y1, x2, y2 = map(int, track["bbox"])
            track_id = track["track_id"]

#            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label_text = f"ID {track_id}"
            
#            cv2.putText(
#                frame,
#                label_text,
#                (x1, y1 - 10),
#                cv2.FONT_HERSHEY_SIMPLEX,
#                0.6,
#                (0, 255, 0),
#                2
#            )
   
        # Draw ball visualization
        ball = entity_manager.ball
        if ball.is_visible():
            ball_position = ball.get_position()
            if ball_position is not None:
                ball_x, ball_y = int(ball_position[0]), int(ball_position[1])
                cv2.circle(frame, (ball_x, ball_y), 6, (0, 140, 255), -1)
                
                trajectory = ball.get_trajectory()
                if len(trajectory) > 1:
                    for i in range(len(trajectory) - 1):
                        pt1 = (int(trajectory[i][0]), int(trajectory[i][1]))
                        pt2 = (int(trajectory[i + 1][0]), int(trajectory[i + 1][1]))
                        cv2.line(frame, pt1, pt2, (0, 180, 255), 2)
        
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