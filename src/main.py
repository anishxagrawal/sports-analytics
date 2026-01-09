# src/main.py

"""
Main orchestration script for sports analytics pipeline.

Connects all pipeline components and runs frame-by-frame processing
on video input with commentary generation.
"""

import cv2

from src.core.video import VideoReader
from src.core.detector import YOLODetector
from src.core.tracker import Tracker
from src.entities.entity_manager import EntityManager
from src.analytics.events import detect_player_events
from src.commentary.engine import CommentaryEngine
from src.commentary.prompt_builder import PromptBuilder
from src.commentary.llm_adapter import LLMAdapter


def main():
    """Run the sports analytics pipeline on a video file."""
    
    # Configuration
    video_path = "data/inputs/test_video.mp4"
    model_path = "yolov8n.pt"
    conf_threshold = 0.5
    allowed_classes = {0}  # Person class
    
    print("Initializing pipeline...")
    
    video_reader = VideoReader(video_path)
    fps = video_reader.fps

    # Output video writer
    output_path = "data/outputs/output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
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
    
    print(f"Processing video at {fps} FPS...\n")
    
    # Frame-by-frame processing loop
    for frame, metadata in video_reader:
        frame_idx = metadata['frame_idx']
        timestamp = metadata['timestamp']
        
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        # Draw tracking results
        for track in tracks:
            x1, y1, x2, y2 = map(int, track["bbox"])
            track_id = track["track_id"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        entity_manager.update(tracks, frame_idx, timestamp)
        
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


if __name__ == "__main__":
    main()
